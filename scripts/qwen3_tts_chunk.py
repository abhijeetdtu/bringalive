#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from scripts.qwen_tts_helpers import (
    load_qwen_model,
    normalize_audio,
    resolve_supported_value,
)
from scripts.tts_helpers import (
    chunk_text,
    cleanup_chunk_dir,
    read_text_with_fallback,
    safe_cache_name,
    stitch_wav_files,
)


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

    try:
        model, _ = load_qwen_model(
            model_name=args.model,
            device_map=args.device_map,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
            torch_module=torch,
            qwen_model_cls=Qwen3TTSModel,
        )
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
    else:
        langs = model.get_supported_languages()

    speakers = model.get_supported_speakers()
    if args.list_speakers:
        print("\n".join(speakers))
        if not args.text and not args.text_file:
            return

    speaker = resolve_supported_value(
        requested_value=args.speaker,
        supported_values=speakers,
        label="speaker",
        default_to_first=True,
    )
    language = resolve_supported_value(
        requested_value=args.language,
        supported_values=langs,
        label="language",
        default_to_first=False,
    )
    print(f"Using speaker: {speaker}")
    print(f"Using language: {language}")

    text = read_text_with_fallback(
        args.text_file, args.text_encoding) if args.text_file else args.text
    text = (text or "").strip()
    if not text:
        raise SystemExit("Empty text")

    chunks = chunk_text(text, args.max_chars)
    if not chunks:
        raise SystemExit("No chunks to synthesize")

    auto_chunk_dir = args.chunk_dir is None
    if args.chunk_dir is not None:
        chunk_root = args.chunk_dir
    else:
        base_name = args.text_file.stem if args.text_file else "input_text"
        chunk_root = Path("app/cache") / safe_cache_name(base_name)
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
                    language=language,
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
                chunk_audio = normalize_audio(wav, np)
                if chunk_audio.size == 0:
                    raise SystemExit(f"No audio generated for chunk {idx}")
                sf.write(tmp_path, chunk_audio, sample_rate, subtype="FLOAT")
            del wavs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    stitch_wav_files(temp_paths, args.out, subtype="PCM_16")
    cleanup_chunk_dir(chunk_root, auto_chunk_dir, args.keep_chunks)

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
    conda run --no-capture-output -n qwen3-tts-cuda python -m scripts.qwen3_tts_chunk `
      --device-map "cuda:0" `
      --dtype "bfloat16" `
      --batch-size 16 `
      --max-chars 350 `
      --max-new-tokens 500 `
      --text-file bringalive\documents\Ramachandra-Guha-Gandhi-Before-India-_2012_-Viking-_India__-libgen.li.txt `
      --speaker "Ryan" `
      --language "English" `
      --chunk-dir "app/cache/gandhi_before_india/qwen3/1" `
      --out "bringalive/documents/gandhi_before_india.wav" `
      --resume `
    
    Stable starting point:
    conda run --no-capture-output -n qwen3-tts-cuda python -u scripts/qwen3_tts_chunk.py --device-map "cuda:0" --dtype "bfloat16" --batch-size 1 --max-new-tokens 256 --text-file "bringalive/documents/short_test.txt" --speaker "Ryan" --language "English" --chunk-dir "app/cache/short_test/qwen3_exp/1" --resume --out "bringalive/documents/short_test_qwen_experiment.wav"

    """
    main()
