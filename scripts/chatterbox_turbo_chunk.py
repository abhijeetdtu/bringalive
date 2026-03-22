#!/usr/bin/env python3
import argparse
import gc
import re
import time
from pathlib import Path

import torch
import soundfile as sf
import tempfile

from bringalive.business_logic.logger import logging
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SR

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
logger = logging.getLogger("scripts.chatterbox_turbo_chunk")


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return parts if parts else [text]


def pack_sentences(sentences, limit):
    """Pack sentences into chunks <= limit chars, maximizing per chunk."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Text to synthesize")
    ap.add_argument("--text-file", type=Path, help="Path to text file")
    ap.add_argument("--out", type=Path, default=Path("turbo_out.wav"))
    ap.add_argument("--device", type=str, default=None, help="mps or cpu")
    ap.add_argument("--max-chars", type=int, default=350)
    ap.add_argument("--voice", type=Path, default=None,
                    help="Optional reference wav")
    ap.add_argument("--chunk-dir", type=Path, default=None,
                    help="Directory to store per-chunk wavs (for resume)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip chunks that already exist in chunk-dir")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="Keep per-chunk wavs (no temp cleanup)")
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
        text = args.text_file.read_text()
    else:
        text = args.text

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info("Config: %s", vars(args))
    logger.info("Device: %s", device)

    model = ChatterboxTurboTTS.from_pretrained(device=device)

    if args.voice:
        model.prepare_conditionals(
            str(args.voice),
            exaggeration=args.exaggeration)

    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, args.max_chars)

    temp_paths = []
    tmp_dir = None
    if args.chunk_dir is not None:
        chunk_root = args.chunk_dir
        chunk_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="chatterbox_turbo_")
        chunk_root = Path(tmp_dir.name)

    use_mps = device == "mps"
    for i, chunk in enumerate(chunks, 1):
        tmp_path = chunk_root / f"chunk_{i:04d}.wav"
        if args.resume and tmp_path.exists():
            print(f"[{i}/{len(chunks)}] exists, skipping")
            temp_paths.append(tmp_path)
            continue

        print(f"[{i}/{len(chunks)}] {len(chunk)} chars")
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.float16):
                wav = model.generate(
                    chunk,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                wav_np = wav.squeeze().detach().cpu().numpy()
        sf.write(tmp_path, wav_np, S3GEN_SR)
        temp_paths.append(tmp_path)

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

    # Stream-append all chunk files to output to avoid high RAM usage
    with sf.SoundFile(args.out, mode="w", samplerate=S3GEN_SR, channels=1) as out_f:
        for p in temp_paths:
            with sf.SoundFile(p, mode="r") as in_f:
                while True:
                    data = in_f.read(8192, dtype="float32")
                    if len(data) == 0:
                        break
                    out_f.write(data)

    if tmp_dir is not None and not args.keep_chunks:
        tmp_dir.cleanup()
    print(f"Saved: {args.out}")


if __name__ == "__main__":

    """
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


    """
    main()
