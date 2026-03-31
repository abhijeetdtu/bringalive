#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VoiceSegment:
    text: str
    tone_description: str
    character: str | None = None


def normalize_audio(wav, np_module):
    audio = np_module.asarray(wav, dtype="float32").reshape(-1)
    if audio.size == 0:
        raise SystemExit("Qwen returned zero-length audio")
    return audio


def resolve_dtype(name: str, torch_module, device_map: str):
    if name == "float32":
        return torch_module.float32
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    if "cuda" in device_map and torch_module.cuda.is_available():
        return (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )
    return torch_module.float32


def configure_torch_for_inference(torch_module, device_map: str):
    if "cuda" not in device_map or not torch_module.cuda.is_available():
        return
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True
    try:
        torch_module.set_float32_matmul_precision("high")
    except RuntimeError:
        pass


def load_qwen_model(
    model_name: str,
    device_map: str,
    dtype_name: str,
    attn_implementation: str | None,
    torch_module,
    qwen_model_cls,
):
    configure_torch_for_inference(torch_module, device_map)
    load_kwargs = {
        "device_map": device_map,
        "dtype": resolve_dtype(dtype_name, torch_module, device_map),
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    print(f"Loading model: {model_name}")
    print(f"device_map={device_map}, dtype={load_kwargs['dtype']}")
    if attn_implementation:
        print(f"attn_implementation={attn_implementation}")
    model = qwen_model_cls.from_pretrained(model_name, **load_kwargs)
    return model, load_kwargs


def _normalize_tone_description(value) -> str:
    if value is None:
        return "neutral voice"
    if isinstance(value, str):
        text = value.strip()
        return text or "neutral voice"
    if isinstance(value, dict):
        parts = []
        ordered_keys = [
            "emotion",
            "tone",
            "speed",
            "volume",
            "pitch",
            "personality",
        ]
        seen = set()
        for key in ordered_keys:
            if key in value and value[key]:
                parts.append(f"{key}: {str(value[key]).strip()}")
                seen.add(key)
        for key, item in value.items():
            if key in seen or not item:
                continue
            parts.append(f"{key}: {str(item).strip()}")
        return ". ".join(parts) if parts else "neutral voice"
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ". ".join(parts) if parts else "neutral voice"
    text = str(value).strip()
    return text or "neutral voice"


def read_ollama_segments(chunk_path: Path):
    data = json.loads(chunk_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of segments in {chunk_path}")

    segments = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"Segment {idx} in {chunk_path} is not an object")
        text = str(item.get("text", "")).strip()
        tone = _normalize_tone_description(item.get("tone_description"))
        character = item.get("character")
        if not text:
            continue
        segments.append(
            VoiceSegment(
                text=text,
                tone_description=tone,
                character=str(character).strip() if character is not None else None,
            )
        )
    return segments


def build_instruct(voice_prompt_prefix: str | None, tone_description: str) -> str:
    tone_description = tone_description.strip()
    prefix = (voice_prompt_prefix or "").strip()
    if prefix and tone_description:
        return f"{prefix}\n\nTone description: {tone_description}"
    if prefix:
        return prefix
    return tone_description


def build_customvoice_instruct(tone_description: str) -> str:
    tone_description = (tone_description or "").strip()
    if not tone_description:
        return ""
    return (
        "Speak this line naturally with the following style guidance: "
        f"{tone_description}"
    )


def collect_ollama_chunk_files(input_dir: Path, pattern: str):
    chunk_files = sorted(input_dir.glob(pattern))
    if not chunk_files:
        raise SystemExit(f"No files matching {pattern!r} found in {input_dir}")
    return chunk_files


def default_ollama_chunk_dir(input_dir: Path, suffix: str):
    return input_dir / suffix


def default_ollama_output_path(input_dir: Path, suffix: str):
    return input_dir / f"{input_dir.name}_{suffix}.wav"


def read_voice_prompt_prefix(path: Path | None):
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Voice prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def resolve_supported_value(
    requested_value: str | None,
    supported_values,
    label: str,
    default_to_first: bool = False,
):
    supported_values = list(supported_values or [])
    if not supported_values:
        if requested_value is None and default_to_first:
            raise SystemExit(f"Model did not return any supported {label}s.")
        return requested_value

    if requested_value is None:
        if default_to_first:
            return supported_values[0]
        return None

    requested_normalized = requested_value.strip().lower()
    for value in supported_values:
        if value.strip().lower() == requested_normalized:
            return value

    supported_preview = ", ".join(str(v) for v in supported_values[:20])
    raise SystemExit(
        f"Unsupported {label}: {requested_value!r}. "
        f"Supported {label}s include: {supported_preview}"
    )


def add_ollama_chunk_input_args(parser):
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Ollama chunk_*.txt files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Final stitched WAV output path",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Directory to store per-chunk WAV files",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="chunk_*.txt",
        help="Glob for Ollama chunk files inside input-dir",
    )
    parser.add_argument(
        "--voice-prompt-file",
        type=Path,
        default=None,
        help="Optional prompt file to prepend before each tone description",
    )


def add_qwen_generation_args(
    parser,
    *,
    default_model: str,
    language_help: str,
    batch_help: str,
):
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="HF model id or local model path",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help=language_help,
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Model device_map (auto, cpu, cuda:0, mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype for from_pretrained",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention backend (e.g. sdpa, flash_attention_2)",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (do_sample=False)",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=batch_help,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunk WAVs that already exist in chunk-dir",
    )


def prepare_ollama_chunk_run(args, *, chunk_dir_suffix: str, out_suffix: str):
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")

    chunk_files = collect_ollama_chunk_files(args.input_dir, args.glob)
    chunk_dir = args.chunk_dir or default_ollama_chunk_dir(args.input_dir, chunk_dir_suffix)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or default_ollama_output_path(args.input_dir, out_suffix)
    voice_prompt_prefix = read_voice_prompt_prefix(args.voice_prompt_file)
    return chunk_files, chunk_dir, out_path, voice_prompt_prefix


def customvoice_ignores_instruct(model) -> bool:
    model_size = str(getattr(model.model, "tts_model_size", "")).strip().lower()
    model_type = str(getattr(model.model, "tts_model_type", "")).strip().lower()
    if model_type != "custom_voice":
        return False
    return model_size in {"0.6b", "0b6", "0.6"}


def estimate_max_new_tokens(
    texts,
    hard_cap: int,
    *,
    min_tokens: int = 96,
    max_auto_tokens: int = 384,
    tokens_per_word: int = 14,
):
    texts = list(texts)
    if not texts:
        return hard_cap

    max_words = 1
    for text in texts:
        words = len(str(text).split())
        if words > max_words:
            max_words = words

    estimated = max(min_tokens, max_words * tokens_per_word)
    estimated = min(estimated, max_auto_tokens)
    return min(hard_cap, estimated)
