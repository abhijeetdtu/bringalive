#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import soundfile as sf

from chatterbox.tts_turbo import S3GEN_SR


def _safe_source_dir(source_id: str) -> str:
    safe = source_id.strip().replace("/", "_").replace("\\", "_")
    safe = safe.replace("..", "_")
    return safe or "default"


def main():
    ap = argparse.ArgumentParser(
        description="Stitch cached TTS chunks into a single WAV.")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("app/cache"),
        help="Cache root directory",
    )
    ap.add_argument(
        "--source-id",
        type=str,
        required=True,
        help="Source ID directory to stitch (same as used for caching)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("stitched.wav"),
        help="Output WAV path",
    )
    args = ap.parse_args()

    source_dir = args.cache_dir / _safe_source_dir(args.source_id)
    if not source_dir.exists():
        raise SystemExit(f"Source cache dir not found: {source_dir}")

    chunks = sorted(source_dir.glob("chunk_*.wav"))
    if not chunks:
        raise SystemExit(f"No cached chunks found in: {source_dir}")

    with sf.SoundFile(args.out, mode="w", samplerate=S3GEN_SR, channels=1) as out_f:
        for p in chunks:
            with sf.SoundFile(p, mode="r") as in_f:
                while True:
                    data = in_f.read(8192, dtype="float32")
                    if len(data) == 0:
                        break
                    out_f.write(data)

    print(f"Stitched {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
