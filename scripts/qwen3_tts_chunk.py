#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return parts if parts else [text]


def pack_sentences(sentences, limit):
    chunks = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not buf:
            if len(s) <= limit:
                buf = s
            else:
                for i in range(0, len(s), limit):
                    chunks.append(s[i: i + limit])
        else:
            candidate = f"{buf} {s}"
            if len(candidate) <= limit:
                buf = candidate
            else:
                chunks.append(buf)
                if len(s) <= limit:
                    buf = s
                else:
                    for i in range(0, len(s), limit):
                        chunks.append(s[i: i + limit])
                    buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _safe_cache_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "input_text"


def _read_text_with_fallback(path: Path, encoding: str | None) -> str:
    if encoding:
        return path.read_text(encoding=encoding)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_audio(wav):
    audio = np.asarray(wav, dtype="float32").reshape(-1)
    if audio.size == 0:
        return np.zeros(0, dtype="float32")
    return audio


def _resolve_dtype(name: str, device_map: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if "cuda" in device_map and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _configure_torch_for_inference(device_map: str):
    if "cuda" not in device_map or not torch.cuda.is_available():
        return
    # Favor fast matmul kernels during pure inference workloads.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except RuntimeError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Text to synthesize")
    ap.add_argument("--text-file", type=Path, help="Path to text file")
    ap.add_argument("--text-encoding", type=str, default=None,
                    help="Optional input file encoding (e.g. utf-8, cp1252)")
    ap.add_argument("--out", type=Path, default=Path("qwen3_out.wav"))

    ap.add_argument("--model", type=str,
                    default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    help="HF model id or local model path")
    ap.add_argument("--speaker", type=str, default=None,
                    help="Qwen speaker id (default: first supported speaker)")
    ap.add_argument("--language", type=str, default="english",
                    help="Language code (default: en)")
    ap.add_argument("--instruct", type=str, default=None,
                    help="Optional speaking style instruction")

    ap.add_argument("--device-map", type=str, default="auto",
                    help="Model device_map (auto, cpu, cuda:0, mps)")
    ap.add_argument("--dtype", type=str, default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"],
                    help="Model dtype for from_pretrained")
    ap.add_argument("--attn-implementation", type=str, default=None,
                    help="Optional attention backend (e.g. flash_attention_2)")

    ap.add_argument("--max-chars", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Number of text chunks to synthesize per model call")
    ap.add_argument("--chunk-dir", type=Path, default=None,
                    help="Directory to store per-chunk wavs (for resume)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip chunks that already exist in chunk-dir")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="Keep per-chunk wavs (no auto cleanup)")

    ap.add_argument("--no-sample", action="store_true",
                    help="Disable sampling (do_sample=False)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=2048)

    ap.add_argument("--list-speakers", action="store_true",
                    help="Print supported speakers and exit")
    ap.add_argument("--list-languages", action="store_true",
                    help="Print supported languages and exit")
    args = ap.parse_args()

    if not args.list_speakers and not args.list_languages and not args.text and not args.text_file:
        ap.error("Provide --text or --text-file")

    load_kwargs = {"device_map": args.device_map}
    load_kwargs["dtype"] = _resolve_dtype(args.dtype, args.device_map)
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    _configure_torch_for_inference(args.device_map)

    print(f"Loading model: {args.model}")
    print(f"device_map={args.device_map}, dtype={load_kwargs['dtype']}")
    try:
        model = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)
    except OSError as exc:
        raise SystemExit(
            "Failed to load model.\n"
            f"Requested: {args.model}\n"
            "Try one of:\n"
            "  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign\n"
            "  - Qwen/Qwen3-TTS-12Hz-1.7B-Base\n"
            "If the repo is gated/private, run `hf auth login` first."
        ) from exc

    if args.list_languages:
        langs = model.get_supported_languages()
        print("\n".join(langs))
        if not args.list_speakers:
            return

    speakers = model.get_supported_speakers()
    if args.list_speakers:
        print("\n".join(speakers))
        if not args.text and not args.text_file:
            return

    speaker = args.speaker
    if speaker is None:
        if not speakers:
            raise SystemExit(
                "Model did not return any supported speakers; pass --speaker explicitly.")
        speaker = speakers[0]
    print(f"Using speaker: {speaker}")
    print(f"Using language: {args.language}")

    text = _read_text_with_fallback(
        args.text_file, args.text_encoding) if args.text_file else args.text
    text = (text or "").strip()
    if not text:
        raise SystemExit("Empty text")

    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, args.max_chars)
    if not chunks:
        raise SystemExit("No chunks to synthesize")

    auto_chunk_dir = args.chunk_dir is None
    if args.chunk_dir is not None:
        chunk_root = args.chunk_dir
    else:
        base_name = args.text_file.stem if args.text_file else "input_text"
        chunk_root = Path("app/cache") / _safe_cache_name(base_name)
    chunk_root.mkdir(parents=True, exist_ok=True)

    total = len(chunks)
    temp_paths = [
        chunk_root / f"chunk_{i:04d}.wav" for i in range(1, total + 1)]
    pending_indices = []
    pending_texts = []
    for i, chunk in enumerate(chunks, 1):
        tmp_path = temp_paths[i - 1]
        if args.resume and tmp_path.exists():
            print(f"[{i}/{total}] exists, skipping")
            continue
        print(f"[{i}/{total}] queued ({len(chunk)} chars)")
        pending_indices.append(i)
        pending_texts.append(chunk)

    if pending_texts:
        batch_size = max(1, args.batch_size)
        print(
            f"Running inference for {len(pending_texts)} chunks with batch_size={batch_size}")
        sample_rate = None
        for start in range(0, len(pending_texts), batch_size):
            batch_indices = pending_indices[start:start + batch_size]
            batch_texts = pending_texts[start:start + batch_size]
            print(
                f"Generating batch {start // batch_size + 1} "
                f"({len(batch_texts)} chunk{'s' if len(batch_texts) != 1 else ''})"
            )
            with torch.inference_mode():
                wavs, batch_sample_rate = model.generate_custom_voice(
                    text=batch_texts,
                    speaker=speaker,
                    language=args.language,
                    instruct=args.instruct,
                    non_streaming_mode=True,
                    do_sample=not args.no_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                )
            if len(wavs) != len(batch_texts):
                raise SystemExit(
                    f"Batch output mismatch: got {len(wavs)} wavs for {len(batch_texts)} input chunks")
            if sample_rate is None:
                sample_rate = batch_sample_rate
            elif batch_sample_rate != sample_rate:
                raise SystemExit(
                    f"Sample-rate mismatch across batches: {batch_sample_rate} != {sample_rate}")
            for idx, wav in zip(batch_indices, wavs):
                tmp_path = temp_paths[idx - 1]
                chunk_audio = _normalize_audio(wav)
                if chunk_audio.size == 0:
                    raise SystemExit(f"No audio generated for chunk {idx}")
                sf.write(tmp_path, chunk_audio, sample_rate, subtype="FLOAT")
            del wavs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not temp_paths:
        raise SystemExit("No audio generated")

    first_sr = None
    with sf.SoundFile(temp_paths[0], mode="r") as first_f:
        first_sr = first_f.samplerate

    with sf.SoundFile(args.out, mode="w", samplerate=first_sr, channels=1, subtype="PCM_16") as out_f:
        for p in temp_paths:
            with sf.SoundFile(p, mode="r") as in_f:
                if in_f.samplerate != first_sr:
                    raise SystemExit(
                        f"Sample-rate mismatch in {p}: {in_f.samplerate} != {first_sr}")
                while True:
                    data = in_f.read(8192, dtype="float32")
                    if len(data) == 0:
                        break
                    out_f.write(data)

    if auto_chunk_dir and not args.keep_chunks:
        shutil.rmtree(chunk_root, ignore_errors=True)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    """
    conda activate qwen3-tts

    conda run --name qwen3-tts python -m scripts.qwen3_tts_chunk \
      --text-file bringalive/documents/short_test.txt \
      --speaker Ryan \
      --language English \
      --chunk-dir app/cache/short_test/qwen3/1 \
      --resume \
      --out bringalive/documents/short_test_qwen.wav

    Windows PowerShell:
    conda run --no-capture-output -n qwen3-tts-cuda python -u scripts/qwen3_tts_chunk.py `
      --device-map "cuda:0" `
      --attn-implementation "sdpa" `
      --dtype "float16" `
      --batch-size 1 `
      --no-sample `
      --max-new-tokens 1024 `
      --text-file "bringalive/documents/long_test.txt" `
      --speaker "Ryan" `
      --language "English" `
      --chunk-dir "app/cache/short_test/qwen3/1" `
      --resume `
      --out "bringalive/documents/long_test_qwen.wav"
    
    conda run --no-capture-output -n qwen3-tts-experiment python -u scripts/qwen3_tts_chunk.py --device-map "cuda:0" --attn-implementation "flash_attention_2" --dtype "float16" --batch-size 1 --no-sample --max-new-tokens 1024 --text-file "bringalive/documents/short_test.txt" --speaker "Ryan" --language "English" --chunk-dir "app/cache/short_test/qwen3_exp/1" --resume --out "bringalive/documents/short_test_qwen_experiment.wav"

    """
    main()
