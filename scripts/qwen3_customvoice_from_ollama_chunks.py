#!/usr/bin/env python3
import argparse

from scripts.qwen_tts_helpers import (
    add_ollama_chunk_input_args,
    add_qwen_generation_args,
    build_customvoice_instruct,
    customvoice_ignores_instruct,
    estimate_max_new_tokens,
    load_qwen_model,
    normalize_audio,
    prepare_ollama_chunk_run,
    read_ollama_segments,
    resolve_supported_value,
)
from scripts.tts_helpers import stitch_wav_files


np = None
sf = None
torch = None
Qwen3TTSModel = None


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Ollama chunk JSON files into stitched Qwen CustomVoice audio "
            "while keeping a constant speaker."
        )
    )
    add_ollama_chunk_input_args(parser)
    add_qwen_generation_args(
        parser,
        default_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        language_help="Language passed to Qwen CustomVoice",
        batch_help="Number of Ollama segments to synthesize per CustomVoice model call",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Qwen speaker id; defaults to the first supported speaker",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print supported speakers and exit",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print supported languages and exit",
    )
    parser.add_argument(
        "--disable-auto-max-new-tokens",
        action="store_true",
        help="Use the fixed --max-new-tokens value instead of per-batch auto-capping",
    )
    parser.add_argument(
        "--prepend-voice-prompt",
        action="store_true",
        help="Prepend voice_prompt.txt to CustomVoice instructions. Disabled by default because it can conflict with fixed speaker identity.",
    )
    parser.add_argument(
        "--use-instruct",
        action="store_true",
        help="Enable CustomVoice instruction prompts from tone_description. Disabled by default to match the simpler known-good CustomVoice path.",
    )
    return parser


def main():
    global np, sf, torch, Qwen3TTSModel
    args = build_parser().parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model, _ = load_qwen_model(
        model_name=args.model,
        device_map=args.device_map,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        torch_module=torch,
        qwen_model_cls=Qwen3TTSModel,
    )

    supported_languages = model.get_supported_languages()
    if args.list_languages:
        print("\n".join(supported_languages))
        if not args.list_speakers:
            return

    speakers = model.get_supported_speakers()
    if args.list_speakers:
        print("\n".join(speakers))
        return

    speaker = resolve_supported_value(
        requested_value=args.speaker,
        supported_values=speakers,
        label="speaker",
        default_to_first=True,
    )
    language = resolve_supported_value(
        requested_value=args.language,
        supported_values=supported_languages,
        label="language",
        default_to_first=False,
    )

    chunk_files, chunk_dir, out_path, voice_prompt_prefix = prepare_ollama_chunk_run(
        args,
        chunk_dir_suffix="qwen3_customvoice_tts",
        out_suffix="qwen3_customvoice",
    )
    if customvoice_ignores_instruct(model):
        print(
            "Warning: this CustomVoice model ignores instruction prompts. "
            "voice_prompt.txt and tone_description text will not affect output."
        )

    print(f"Using speaker: {speaker}")
    print(f"Using language: {language}")
    if not args.use_instruct:
        print(
            "CustomVoice instructions are disabled by default. "
            "Using fixed speaker identity with plain text only."
        )
    elif voice_prompt_prefix and not args.prepend_voice_prompt:
        print(
            "Ignoring voice_prompt.txt for CustomVoice by default. "
            "The fixed speaker already defines core voice identity."
        )

    wav_paths = []
    sample_rate = None
    total_files = len(chunk_files)
    for file_idx, chunk_file in enumerate(chunk_files, start=1):
        wav_path = chunk_dir / f"{chunk_file.stem}.wav"
        wav_paths.append(wav_path)
        if args.resume and wav_path.exists():
            print(f"[{file_idx}/{total_files}] exists, skipping {chunk_file.name}")
            continue

        segments = read_ollama_segments(chunk_file)
        if not segments:
            raise SystemExit(f"No valid segments found in {chunk_file}")

        print(
            f"[{file_idx}/{total_files}] {chunk_file.name}: "
            f"{len(segments)} segment{'s' if len(segments) != 1 else ''}"
        )
        chunk_audio = []
        for start in range(0, len(segments), args.batch_size):
            batch_segments = segments[start:start + args.batch_size]
            batch_texts = [segment.text for segment in batch_segments]
            batch_instructs = None
            if args.use_instruct:
                batch_instructs = [
                    build_customvoice_instruct(segment.tone_description)
                    for segment in batch_segments
                ]
                if voice_prompt_prefix and args.prepend_voice_prompt:
                    batch_instructs = [
                        f"{voice_prompt_prefix}\n\n{instruct}".strip()
                        for instruct in batch_instructs
                    ]
            batch_max_new_tokens = args.max_new_tokens
            if not args.disable_auto_max_new_tokens:
                batch_max_new_tokens = estimate_max_new_tokens(
                    batch_texts,
                    hard_cap=args.max_new_tokens,
                )

            for seg_offset, segment in enumerate(batch_segments, start=1):
                seg_idx = start + seg_offset
                print(
                    f"  - segment {seg_idx}/{len(segments)} "
                    f"tone={segment.tone_description!r} chars={len(segment.text)}"
                )
            print(
                f"    batch max_new_tokens={batch_max_new_tokens}"
            )

            with torch.inference_mode():
                wavs, generated_sample_rate = model.generate_custom_voice(
                    text=batch_texts,
                    speaker=speaker,
                    language=language,
                    instruct=batch_instructs,
                    non_streaming_mode=True,
                    do_sample=not args.no_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=batch_max_new_tokens,
                )

            if len(wavs) != len(batch_segments):
                raise SystemExit(
                    f"Expected {len(batch_segments)} wavs for batch in {chunk_file.name}, "
                    f"got {len(wavs)}"
                )
            print(
                f"    generated batch "
                f"{start // args.batch_size + 1} "
                f"({len(batch_segments)} segment{'s' if len(batch_segments) != 1 else ''})"
            )
            if sample_rate is None:
                sample_rate = generated_sample_rate
            elif generated_sample_rate != sample_rate:
                raise SystemExit(
                    f"Sample-rate mismatch: {generated_sample_rate} != {sample_rate}"
                )
            for wav in wavs:
                chunk_audio.append(normalize_audio(wav, np))

        sf.write(
            wav_path,
            np.concatenate(chunk_audio).astype("float32", copy=False),
            sample_rate,
            subtype="FLOAT",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stitch_wav_files(wav_paths, out_path, samplerate=sample_rate, subtype="PCM_16")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    r"""
    Windows PowerShell:
    conda run --no-capture-output -n qwen3-tts-cuda python -m scripts.qwen3_customvoice_from_ollama_chunks `
      --input-dir "app\cache\ollama\long_test" `
      --glob "chunk_*.txt" `
      --voice-prompt-file "extensions\openwebui_chunk_reader\voice_prompt.txt" `
      --chunk-dir "app\cache\long_test\qwen3_customvoice\1" `
      --out "bringalive\documents\long_test_qwen_customvoice.wav" `
      --model "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" `
      --language "English" `
      --speaker "Ryan" `
      --device-map "cuda:0" `
      --dtype "bfloat16" `
      --batch-size 16 `
      --max-new-tokens 256 `
      --resume

    To try the more experimental instruction-conditioned path:
      add --use-instruct
      optionally add --prepend-voice-prompt
    """
    main()
