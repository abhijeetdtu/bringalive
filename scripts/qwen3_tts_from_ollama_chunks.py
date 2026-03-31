#!/usr/bin/env python3
import argparse

from scripts.qwen_tts_helpers import (
    add_ollama_chunk_input_args,
    add_qwen_generation_args,
    build_instruct,
    load_qwen_model,
    normalize_audio,
    prepare_ollama_chunk_run,
    read_ollama_segments,
)
from scripts.tts_helpers import stitch_wav_files


np = None
sf = None
torch = None
Qwen3TTSModel = None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert Ollama chunk JSON files into stitched Qwen TTS audio."
    )
    add_ollama_chunk_input_args(parser)
    add_qwen_generation_args(
        parser,
        default_model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        language_help="Language passed to Qwen voice design",
        batch_help="Number of Ollama segments to synthesize per VoiceDesign model call",
    )
    return parser


def main():
    global np, sf, torch, Qwen3TTSModel
    args = build_parser().parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    chunk_files, chunk_dir, out_path, voice_prompt_prefix = prepare_ollama_chunk_run(
        args,
        chunk_dir_suffix="qwen3_tts",
        out_suffix="qwen3",
    )

    model, _ = load_qwen_model(
        model_name=args.model,
        device_map=args.device_map,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        torch_module=torch,
        qwen_model_cls=Qwen3TTSModel,
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
            batch_instructs = [
                build_instruct(voice_prompt_prefix, segment.tone_description)
                for segment in batch_segments
            ]
            batch_languages = [args.language] * len(batch_segments)

            for seg_offset, segment in enumerate(batch_segments, start=1):
                seg_idx = start + seg_offset
                print(
                    f"  - segment {seg_idx}/{len(segments)} "
                    f"tone={segment.tone_description!r} chars={len(segment.text)}"
                )

            with torch.inference_mode():
                wavs, generated_sample_rate = model.generate_voice_design(
                    text=batch_texts,
                    language=batch_languages,
                    instruct=batch_instructs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.no_sample,
                    subtalker_dosample=not args.no_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
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
    conda run --no-capture-output -n qwen3-tts-cuda python -m scripts.qwen3_tts_from_ollama_chunks `
      --input-dir "app\cache\ollama\short_test" `
      --glob "chunk_*.txt" `
      --voice-prompt-file "extensions\openwebui_chunk_reader\voice_prompt.txt" `
      --chunk-dir "app\cache\short_test\qwen3_voice_design\1" `
      --out "bringalive\documents\short_test_qwen_voice_design.wav" `
      --model "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" `
      --language "English" `
      --device-map "cuda:0" `
      --dtype "bfloat16" `
      --batch-size 1 `
      --max-new-tokens 256 `
      --resume
    """
    main()
