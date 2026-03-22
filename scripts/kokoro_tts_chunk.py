#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from soundfile import LibsndfileError

from kokoro import KPipeline

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
DEFAULT_SAMPLE_RATE = 24000


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


def _to_audio_np(result) -> np.ndarray:
    if result.output is None or result.output.audio is None:
        raise ValueError("kokoro returned empty audio output")
    audio = result.output.audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype="float32").reshape(-1)
    if audio.size == 0:
        raise ValueError("kokoro returned zero-length audio")
    return audio


def _safe_cache_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "input_text"


def _bytes_per_sample(subtype: str) -> int:
    mapping = {
        "PCM_U8": 1,
        "PCM_16": 2,
        "PCM_24": 3,
        "PCM_32": 4,
        "FLOAT": 4,
        "DOUBLE": 8,
    }
    return mapping.get(subtype.upper(), 2)


def _read_text_with_fallback(path: Path, encoding: str | None) -> str:
    if encoding:
        return path.read_text(encoding=encoding)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last resort: preserve progress by replacing bad bytes.
    return path.read_text(encoding="utf-8", errors="replace")


def _part_path(base_out: Path, part_idx: int) -> Path:
    return base_out.with_name(
        f"{base_out.stem}_part{part_idx:03d}{base_out.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Text to synthesize")
    ap.add_argument("--text-file", type=Path, help="Path to text file")
    ap.add_argument("--text-encoding", type=str, default=None,
                    help="Optional input file encoding (e.g. utf-8, cp1252)")
    ap.add_argument("--out", type=Path, default=Path("kokoro_out.wav"))
    ap.add_argument("--lang-code", type=str, default="a",
                    help="Kokoro language code (default: a)")
    ap.add_argument("--voice", type=str, default="af_heart",
                    help="Kokoro voice id (default: af_heart)")
    ap.add_argument(
        "--voice-clone-pack",
        type=Path,
        default=None,
        help="Path to cloned Kokoro voice pack (.pt). Overrides --voice unless --voice-mix is set.",
    )
    ap.add_argument(
        "--voice-mix",
        type=str,
        default=None,
        help="Comma-separated voices/packs to blend, e.g. 'af_heart,/path/clone.pt'.",
    )
    ap.add_argument(
        "--speed",
        "--pipeline-speed",
        dest="speed",
        type=float,
        default=1.0,
        help="Speed passed directly to KPipeline(...) calls",
    )
    ap.add_argument("--max-chars", type=int, default=350)
    ap.add_argument("--split-pattern", type=str, default=None,
                    help="Optional regex split pattern passed to KPipeline")
    ap.add_argument("--repo-id", type=str, default="hexgrad/Kokoro-82M",
                    help="Hugging Face repo id for model assets")
    ap.add_argument("--device", type=str, default=None,
                    help="Device override passed to KPipeline (e.g. cpu, mps)")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument(
        "--out-subtype",
        type=str,
        default="PCM_16",
        help="Final output subtype (e.g. PCM_16, PCM_24, FLOAT). Default: PCM_16",
    )
    ap.add_argument(
        "--out-format",
        type=str,
        default=None,
        help="Optional final output container format override (e.g. WAV, RF64, FLAC).",
    )
    ap.add_argument("--chunk-dir", type=Path, default=None,
                    help="Directory to store per-chunk wavs (for resume)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip chunks that already exist in chunk-dir")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="Keep per-chunk wavs (no temp cleanup)")
    args = ap.parse_args()

    if not args.text and not args.text_file:
        ap.error("Provide --text or --text-file")

    text = _read_text_with_fallback(
        args.text_file, args.text_encoding) if args.text_file else args.text
    text = (text or "").strip()
    if not text:
        raise SystemExit("Empty text")

    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, args.max_chars)
    if not chunks:
        raise SystemExit("No chunks to synthesize")

    if args.voice_clone_pack is not None and args.voice_clone_pack.suffix != ".pt":
        raise SystemExit("--voice-clone-pack must point to a .pt file")

    if args.voice_clone_pack is not None and not args.voice_clone_pack.exists():
        raise SystemExit(
            f"voice clone pack not found: {args.voice_clone_pack}")
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")

    if args.voice_mix and args.voice_clone_pack:
        clone_pack = str(args.voice_clone_pack)
        voice_selector = f"{args.voice_mix},{clone_pack}"
    elif args.voice_mix:
        voice_selector = args.voice_mix
    elif args.voice_clone_pack:
        voice_selector = str(args.voice_clone_pack)
    else:
        voice_selector = args.voice

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using voice: {voice_selector}")
    print(f"Using speed: {args.speed}")

    pipeline = KPipeline(
        lang_code=args.lang_code,
        repo_id=args.repo_id,
        device=device,
    )

    temp_paths = []
    generated_paths = []
    auto_chunk_dir = args.chunk_dir is None
    if args.chunk_dir is not None:
        chunk_root = args.chunk_dir
    else:
        base_name = args.text_file.stem if args.text_file else "input_text"
        chunk_root = Path("app/cache") / _safe_cache_name(base_name)
    chunk_root.mkdir(parents=True, exist_ok=True)

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        tmp_path = chunk_root / f"chunk_{i:04d}.wav"
        if args.resume and tmp_path.exists():
            print(f"[{i}/{total}] exists, skipping")
            temp_paths.append(tmp_path)
            continue
        print(f"[{i}/{total}] {len(chunk)} chars")
        generator = pipeline(
            chunk,
            voice=voice_selector,
            speed=args.speed,
            split_pattern=args.split_pattern,
        )
        chunk_audio = []
        for result in generator:
            chunk_audio.append(_to_audio_np(result))

        if not chunk_audio:
            raise SystemExit(f"No audio generated for chunk {i}")
        combined_chunk = (
            chunk_audio[0]
            if len(chunk_audio) == 1
            else np.concatenate(chunk_audio).astype("float32", copy=False)
        )
        sf.write(tmp_path, combined_chunk, args.sample_rate, subtype="FLOAT")
        temp_paths.append(tmp_path)
        generated_paths.append(tmp_path)

    if not temp_paths:
        raise SystemExit("No audio generated")

    out_format = args.out_format
    total_frames = 0
    for p in temp_paths:
        info = sf.info(p)
        total_frames += int(info.frames)

    if out_format is None and args.out.suffix.lower() == ".wav":
        estimated = total_frames * _bytes_per_sample(args.out_subtype) + 44
        wav_limit = 0xFFFFFFFF
        if estimated >= wav_limit:
            bytes_per_sample = _bytes_per_sample(args.out_subtype)
            max_frames_per_file = (wav_limit - 44) // bytes_per_sample
            num_parts = max(1, math.ceil(total_frames / max_frames_per_file))
            print(
                "Large WAV output detected; splitting into "
                f"{num_parts} files to stay within WAV size limits."
            )

            args.out.parent.mkdir(parents=True, exist_ok=True)
            part_idx = 1
            remaining_in_part = max_frames_per_file
            out_path = _part_path(args.out, part_idx)
            out_f = sf.SoundFile(
                out_path,
                mode="w",
                samplerate=args.sample_rate,
                channels=1,
                subtype=args.out_subtype,
                format="WAV",
            )
            try:
                for p in temp_paths:
                    with sf.SoundFile(p, mode="r") as in_f:
                        while True:
                            data = in_f.read(8192, dtype="float32")
                            if len(data) == 0:
                                break
                            offset = 0
                            while offset < len(data):
                                if remaining_in_part == 0:
                                    out_f.close()
                                    part_idx += 1
                                    out_path = _part_path(args.out, part_idx)
                                    out_f = sf.SoundFile(
                                        out_path,
                                        mode="w",
                                        samplerate=args.sample_rate,
                                        channels=1,
                                        subtype=args.out_subtype,
                                        format="WAV",
                                    )
                                    remaining_in_part = max_frames_per_file
                                write_n = min(len(data) - offset,
                                              remaining_in_part)
                                out_f.write(data[offset: offset + write_n])
                                offset += write_n
                                remaining_in_part -= write_n
            except LibsndfileError as e:
                raise SystemExit(
                    f"Failed writing split WAV outputs ({e}). "
                    "Try reducing --out-subtype size or output length."
                ) from e
            finally:
                out_f.close()

            first = _part_path(args.out, 1)
            last = _part_path(args.out, part_idx)
            print(f"Saved: {first} ... {last}")
            if auto_chunk_dir and not args.keep_chunks:
                for p in generated_paths:
                    p.unlink(missing_ok=True)
            return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sf.SoundFile(
            args.out,
            mode="w",
            samplerate=args.sample_rate,
            channels=1,
            subtype=args.out_subtype,
            format=out_format,
        ) as out_f:
            for p in temp_paths:
                with sf.SoundFile(p, mode="r") as in_f:
                    while True:
                        data = in_f.read(8192, dtype="float32")
                        if len(data) == 0:
                            break
                        out_f.write(data)
    except LibsndfileError as e:
        raise SystemExit(
            f"Failed writing output file ({e}). "
            "Try reducing --out-subtype size or output length."
        ) from e

    if auto_chunk_dir and not args.keep_chunks:
        for p in generated_paths:
            p.unlink(missing_ok=True)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    """
    python -m scripts.kokoro_tts_chunk \
      --text-file bringalive/documents/short_test.txt \
      --voice-mix am_michael \
      --speed .7 \
      --max-chars 400 \
      --out bringalive/documents/short_test.wav

     python -m scripts.kokoro_tts_chunk \
        --text-file /Users/abhijeetpokhriyal/Downloads/three_men_in_a_boat.txt \
        --voice af_heart \
        --out bringalive/documents/three_men_in_a_boat.wav \
        --max-chars 1000 \
        --speed .7 \
        --resume
    """
    main()
