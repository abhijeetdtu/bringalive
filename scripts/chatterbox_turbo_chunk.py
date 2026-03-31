#!/usr/bin/env python3
import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import perth
import psutil
import torch
import soundfile as sf
import tempfile

from bringalive.business_logic.logger import logging
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SR
from scripts.tts_helpers import chunk_text, read_text_with_fallback, stitch_wav_files
logger = logging.getLogger("scripts.chatterbox_turbo_chunk")


def _patch_perth_watermarker():
    watermarker_cls = getattr(perth, "PerthImplicitWatermarker", None)
    if watermarker_cls is None:
        logger.warning(
            "PerthImplicitWatermarker is unavailable; falling back to DummyWatermarker."
        )
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
def _sync_device(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _set_process_priority(high_priority: bool):
    if not high_priority:
        return
    try:
        proc = psutil.Process(os.getpid())
        if os.name == "nt":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            logger.info("Process priority set to HIGH_PRIORITY_CLASS")
        else:
            proc.nice(-5)
            logger.info("Raised process priority via niceness")
    except Exception as exc:
        logger.warning("Unable to raise process priority: %s", exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Text to synthesize")
    ap.add_argument("--text-file", type=Path, help="Path to text file")
    ap.add_argument("--out", type=Path, default=Path("turbo_out.wav"))
    ap.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    ap.add_argument("--max-chars", type=int, default=350)
    ap.add_argument("--voice", type=Path, default=None,
                    help="Optional reference wav")
    ap.add_argument("--chunk-dir", type=Path, default=None,
                    help="Directory to store per-chunk wavs (for resume)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip chunks that already exist in chunk-dir")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="Keep per-chunk wavs (no temp cleanup)")
    ap.add_argument("--high-priority", action="store_true",
                    help="Raise process priority during synthesis")
    ap.add_argument(
        "--exaggeration",
        type=float,
        default=0.2,
        help="Exaggeration for prepare_conditionals (0.0-1.0)",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    if not args.text and not args.text_file:
        ap.error("Provide --text or --text-file")

    if args.text_file:
        text = read_text_with_fallback(args.text_file, None)
    else:
        text = args.text

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("Config: %s", vars(args))
    logger.info("Device: %s", device)

    _set_process_priority(args.high_priority)
    _patch_perth_watermarker()
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    if args.voice:
        model.prepare_conditionals(
            str(args.voice),
            exaggeration=args.exaggeration)

    chunks = chunk_text(text, args.max_chars)

    temp_paths = []
    tmp_dir = None
    cleanup_chunk_root = False
    if args.chunk_dir is not None:
        chunk_root = args.chunk_dir
        chunk_root.mkdir(parents=True, exist_ok=True)
        cleanup_chunk_root = True
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="chatterbox_turbo_")
        chunk_root = Path(tmp_dir.name)

    synthesis_wall_seconds = 0.0
    generated_audio_seconds = 0.0
    for i, chunk in enumerate(chunks, 1):
        tmp_path = chunk_root / f"chunk_{i:04d}.wav"
        if args.resume and tmp_path.exists():
            print(f"[{i}/{len(chunks)}] exists, skipping")
            temp_paths.append(tmp_path)
            continue

        print(f"[{i}/{len(chunks)}] {len(chunk)} chars")
        autocast_enabled = device in {"cuda", "mps"}
        _sync_device(device)
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=autocast_enabled):
                wav = model.generate(
                    chunk,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                wav_np = wav.squeeze().detach().cpu().numpy()
        _sync_device(device)
        elapsed = time.perf_counter() - started
        audio_seconds = float(len(wav_np)) / float(S3GEN_SR)
        chunk_rtf = elapsed / audio_seconds if audio_seconds > 0 else float("inf")
        synthesis_wall_seconds += elapsed
        generated_audio_seconds += audio_seconds
        sf.write(tmp_path, wav_np, S3GEN_SR)
        temp_paths.append(tmp_path)
        print(
            f"[{i}/{len(chunks)}] wall={elapsed:.2f}s "
            f"audio={audio_seconds:.2f}s rtf={chunk_rtf:.3f}"
        )

        # Free memory aggressively to avoid MPS/CPU growth over long runs
        # del wav, wav_np
        # if use_mps:
        #     torch.mps.empty_cache()
        # gc.collect()
        if i % 10 == 0:
            time.sleep(5)

    if not temp_paths:
        if tmp_dir is not None and not args.keep_chunks:
            tmp_dir.cleanup()
        raise SystemExit("No audio generated")

    total_output_audio_seconds = 0.0
    for p in temp_paths:
        with sf.SoundFile(p, mode="r") as in_f:
            total_output_audio_seconds += len(in_f) / float(in_f.samplerate)

    stitch_wav_files(temp_paths, args.out, samplerate=S3GEN_SR)

    if not args.keep_chunks:
        if tmp_dir is not None:
            tmp_dir.cleanup()
        elif cleanup_chunk_root:
            shutil.rmtree(chunk_root, ignore_errors=True)
    print(f"Saved: {args.out}")
    if generated_audio_seconds > 0:
        overall_rtf = synthesis_wall_seconds / generated_audio_seconds
        print(
            f"Generated audio: {generated_audio_seconds:.2f}s, "
            f"synthesis wall time: {synthesis_wall_seconds:.2f}s, "
            f"RTF: {overall_rtf:.3f}"
        )
    if total_output_audio_seconds > 0:
        print(f"Final stitched audio duration: {total_output_audio_seconds:.2f}s")


if __name__ == "__main__":

    r"""
    --voice /Users/abhijeetpokhriyal/Downloads/voice_clone.wav \

    python -m scripts.chatterbox_turbo_chunk \
        --text-file bringalive/documents/newport.txt\
        --voice /Users/abhijeetpokhriyal/code/bringalive/app/assets/voices/voice_clone_alan.wav \
        --exaggeration 0.2 \
        --temperature 0.7 \
        --top-p 0.92 \
        --repetition-penalty 1.1 \
        --chunk-dir app/cache/newport/1 \
        --resume \
        --out bringalive/documents/newport.wav

    Windows PowerShell (CUDA):
    conda run --no-capture-output -n chatterbox-tts python -m scripts.chatterbox_turbo_chunk `
        --high-priority `
        --device cuda `
        --voice "app\voices\voice_khalili.wav" `
        --text-file "bringalive\documents\Blue Highways.txt" `
        --chunk-dir "app/cache/short_test/chatterbox/1" `
         --exaggeration 0.35 `
        --temperature 0.7 `
        --top-p 0.92 `
        --repetition-penalty 1.1 `
        --resume `
        --out bringalive/documents/blue_highways.wav

    """
    main()
