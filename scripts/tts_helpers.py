#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

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
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not buf:
            if len(sentence) <= limit:
                buf = sentence
            else:
                for i in range(0, len(sentence), limit):
                    chunks.append(sentence[i: i + limit])
        else:
            candidate = f"{buf} {sentence}"
            if len(candidate) <= limit:
                buf = candidate
            else:
                chunks.append(buf)
                if len(sentence) <= limit:
                    buf = sentence
                else:
                    for i in range(0, len(sentence), limit):
                        chunks.append(sentence[i: i + limit])
                    buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(text: str, max_chars: int):
    return pack_sentences(split_sentences(text), max_chars)


def safe_cache_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "input_text"


def read_text_with_fallback(path: Path, encoding: str | None) -> str:
    if encoding:
        return path.read_text(encoding=encoding)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def stitch_wav_files(
    input_paths,
    out_path: Path,
    samplerate: int | None = None,
    subtype: str | None = None,
    out_format: str | None = None,
):
    import soundfile as sf

    if not input_paths:
        raise SystemExit("No audio generated")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if samplerate is None:
        with sf.SoundFile(input_paths[0], mode="r") as first_f:
            samplerate = first_f.samplerate

    soundfile_kwargs = {
        "mode": "w",
        "samplerate": samplerate,
        "channels": 1,
    }
    if subtype is not None:
        soundfile_kwargs["subtype"] = subtype
    if out_format is not None:
        soundfile_kwargs["format"] = out_format

    with sf.SoundFile(out_path, **soundfile_kwargs) as out_f:
        for path in input_paths:
            with sf.SoundFile(path, mode="r") as in_f:
                if in_f.samplerate != samplerate:
                    raise SystemExit(
                        f"Sample-rate mismatch in {path}: {in_f.samplerate} != {samplerate}"
                    )
                while True:
                    data = in_f.read(8192, dtype="float32")
                    if len(data) == 0:
                        break
                    out_f.write(data)


def cleanup_chunk_dir(chunk_root: Path, auto_chunk_dir: bool, keep_chunks: bool):
    if auto_chunk_dir and not keep_chunks:
        shutil.rmtree(chunk_root, ignore_errors=True)
