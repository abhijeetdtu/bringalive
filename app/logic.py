import gc
import hashlib
import os
import re
import struct
import tempfile
import time
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SR

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
KOKORO_SR = 24000


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


def wav_header_float32(sr: int, channels: int = 1) -> bytes:
    # Stream-friendly WAV header with unknown data size (0xFFFFFFFF)
    bits_per_sample = 32
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sr * block_align
    data_size = 0xFFFFFFFF
    riff_size = 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        3,  # IEEE float
        channels,
        sr,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def synthesize_to_wav_path(
    model: ChatterboxTurboTTS,
    device: str,
    text: str,
    voice_path: str | None,
    exaggeration: float,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_chars: int,
    sleep_every: int,
    sleep_seconds: float,
    speed: float,
    cache_dir: Path | None = None,
    source_id: str | None = None,
) -> Path:
    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")

    if voice_path:
        model.prepare_conditionals(voice_path, exaggeration=exaggeration)

    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, max_chars)
    if not chunks:
        raise ValueError("no chunks to synthesize")

    tmp = tempfile.NamedTemporaryFile(
        prefix="chatterbox_turbo_", suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    with sf.SoundFile(tmp_path, mode="w", samplerate=S3GEN_SR, channels=1) as out_f:
        for i, wav_np in enumerate(
            _generate_chunks(
                model=model,
                device=device,
                text_chunks=chunks,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                speed=speed,
                cache_dir=cache_dir,
                source_id=source_id,
            ),
            1,
        ):
            out_f.write(wav_np)
            if sleep_every > 0 and i % sleep_every == 0:
                time.sleep(sleep_seconds)

    finalize_cached_run(cache_dir, source_id, chunks)
    return tmp_path


def _generate_chunks(
    model: ChatterboxTurboTTS,
    device: str,
    text_chunks: list[str],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    speed: float,
    cache_dir: Path | None = None,
    source_id: str | None = None,
) -> Iterable["np.ndarray"]:
    use_mps = device == "mps"
    for i, chunk in enumerate(text_chunks, 1):
        cached = _load_cached_wav(cache_dir, source_id, chunk, i)
        if cached is not None:
            if speed and speed != 1.0:
                yield _apply_speed(cached, speed)
            else:
                yield cached
            continue
        with torch.inference_mode():
            # Autocast to fp16 on MPS (saves bandwidth / time)
            with torch.autocast(device_type=device, dtype=torch.float8_e4m3fn):
                wav = model.generate(
                    chunk,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
        wav_np = wav.squeeze().detach().cpu().numpy().astype("float32", copy=False)
        _save_cached_wav(cache_dir, source_id, chunk, i, wav_np)
        if speed and speed != 1.0:
            yield _apply_speed(wav_np, speed)
        else:
            yield wav_np

        # del wav, wav_np
        # if use_mps:
        #     torch.mps.empty_cache()
        # gc.collect()


def _apply_speed(wav_np, speed: float):
    # Speed <1.0 slows down, >1.0 speeds up.
    # Preserve pitch when possible. Prefer rubberband if installed.
    if speed <= 0:
        return wav_np
    speed = max(0.5, min(2.0, float(speed)))
    if speed == 1.0:
        return wav_np
    wav_np = wav_np.astype("float32", copy=False)
    try:
        import pyrubberband as pyrb  # type: ignore
        return pyrb.time_stretch(wav_np, S3GEN_SR, speed)
    except Exception:
        try:
            return librosa.effects.time_stretch(wav_np, rate=speed)
        except Exception:
            return wav_np


def stream_wav(
    model: ChatterboxTurboTTS,
    device: str,
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_chars: int,
    start_after_chunks: int,
    progress_cb=None,
    speed: float = 0.9,
    cache_dir: Path | None = None,
    source_id: str | None = None,
) -> Iterable[bytes]:
    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")

    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, max_chars)
    if not chunks:
        raise ValueError("no chunks to synthesize")

    buffered = []
    started = False

    yield wav_header_float32(S3GEN_SR, channels=1)

    total = len(chunks)
    if progress_cb:
        progress_cb(0, total)
    for i, wav_np in enumerate(
        _generate_chunks(
            model=model,
            device=device,
            text_chunks=chunks,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            speed=speed,
            cache_dir=cache_dir,
            source_id=source_id,
        ),
        1,
    ):
        if progress_cb:
            progress_cb(i, total)
        chunk_bytes = wav_np.tobytes()

        if not started:
            buffered.append(chunk_bytes)
            if i >= start_after_chunks:
                for b in buffered:
                    yield b
                buffered.clear()
                started = True
        else:
            yield chunk_bytes

    if buffered:
        for b in buffered:
            yield b

    finalize_cached_run(cache_dir, source_id, chunks)


def _cache_key(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _safe_source_dir(source_id: str) -> str:
    safe = source_id.strip().replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    safe = safe.replace("..", "_")
    return safe or "default"


def _cache_path(
        cache_dir: Path, source_id: str, text: str, chunk_index: int) -> Path:
    key = _cache_key(text)[:10]
    src_dir = _safe_source_dir(source_id)
    filename = f"chunk_{chunk_index:05d}_{key}.wav"
    return cache_dir / src_dir / filename


def _cache_manifest_path(cache_dir: Path, source_id: str) -> Path:
    src_dir = cache_dir / _safe_source_dir(source_id)
    return src_dir / "_latest_manifest.txt"


def _write_cache_manifest(
    cache_dir: Path | None,
    source_id: str | None,
    text_chunks: list[str],
) -> None:
    if not cache_dir or not source_id or not text_chunks:
        return
    try:
        src_dir = cache_dir / _safe_source_dir(source_id)
        src_dir.mkdir(parents=True, exist_ok=True)
        manifest = _cache_manifest_path(cache_dir, source_id)
        lines = [
            _cache_path(cache_dir, source_id, chunk, i).name
            for i, chunk in enumerate(text_chunks, 1)
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        return


def stitch_cached_wav_path(cache_dir: Path, source_id: str) -> Path:
    src_dir = cache_dir / _safe_source_dir(source_id)
    if not src_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {src_dir}")
    manifest = _cache_manifest_path(cache_dir, source_id)
    if not manifest.exists():
        raise FileNotFoundError(
            f"cache manifest not found for source: {source_id}")
    chunks = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        if "/" in name or "\\" in name:
            continue
        p = src_dir / name
        if p.exists():
            chunks.append(p)
    if not chunks:
        raise FileNotFoundError(f"no cached chunks listed in manifest: {manifest}")
    tmp = tempfile.NamedTemporaryFile(
        prefix="chatterbox_stitch_", suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with sf.SoundFile(tmp_path, mode="w", samplerate=S3GEN_SR, channels=1) as out_f:
        for p in chunks:
            with sf.SoundFile(p, mode="r") as in_f:
                while True:
                    data = in_f.read(8192, dtype="float32")
                    if len(data) == 0:
                        break
                    out_f.write(data)
    return tmp_path


def _load_cached_wav(
        cache_dir: Path | None, source_id: str | None, text: str, chunk_index: int):
    if not cache_dir or not source_id:
        return None
    try:
        path = _cache_path(cache_dir, source_id, text, chunk_index)
        if not path.exists():
            return None
        wav_np, _sr = sf.read(path, dtype="float32")
        if wav_np is None:
            return None
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        if wav_np.size == 0:
            return None
        return wav_np.astype("float32", copy=False)
    except Exception:
        return None


def _save_cached_wav(
        cache_dir: Path | None, source_id: str | None, text: str, chunk_index: int, wav_np):
    if not cache_dir or not source_id:
        return
    try:
        path = _cache_path(cache_dir, source_id, text, chunk_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            sf.write(
                path, wav_np.astype("float32", copy=False),
                S3GEN_SR, subtype="FLOAT")
    except Exception:
        return


def finalize_cached_run(
    cache_dir: Path | None,
    source_id: str | None,
    text_chunks: list[str],
) -> None:
    _write_cache_manifest(cache_dir, source_id, text_chunks)


def _kokoro_chunk_audio(pipeline, chunk: str, voice: str, speed: float):
    parts = []
    for result in pipeline(chunk, voice=voice, speed=speed, split_pattern=None):
        if result.output is None or result.output.audio is None:
            continue
        audio = result.output.audio.detach().cpu().numpy().astype("float32", copy=False)
        parts.append(audio.reshape(-1))
    if not parts:
        raise ValueError("kokoro returned no audio")
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _generate_kokoro_chunks(
    pipeline,
    text_chunks: list[str],
    voice: str,
    speed: float,
    cache_dir: Path | None = None,
    source_id: str | None = None,
):
    for i, chunk in enumerate(text_chunks, 1):
        cached = _load_cached_wav(cache_dir, source_id, chunk, i)
        if cached is not None:
            yield cached
            continue
        wav_np = _kokoro_chunk_audio(pipeline, chunk, voice=voice, speed=speed)
        _save_cached_wav(cache_dir, source_id, chunk, i, wav_np)
        yield wav_np


def synthesize_kokoro_to_wav_path(
    pipeline,
    text: str,
    voice: str,
    speed: float,
    max_chars: int,
    sleep_every: int,
    sleep_seconds: float,
    cache_dir: Path | None = None,
    source_id: str | None = None,
) -> Path:
    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")
    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, max_chars)
    if not chunks:
        raise ValueError("no chunks to synthesize")

    tmp = tempfile.NamedTemporaryFile(
        prefix="kokoro_tts_", suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with sf.SoundFile(tmp_path, mode="w", samplerate=KOKORO_SR, channels=1) as out_f:
        for i, wav_np in enumerate(
            _generate_kokoro_chunks(
                pipeline=pipeline,
                text_chunks=chunks,
                voice=voice,
                speed=speed,
                cache_dir=cache_dir,
                source_id=source_id,
            ),
            1,
        ):
            out_f.write(wav_np)
            if sleep_every > 0 and i % sleep_every == 0:
                time.sleep(sleep_seconds)

    finalize_cached_run(cache_dir, source_id, chunks)
    return tmp_path


def stream_kokoro_wav(
    pipeline,
    text: str,
    voice: str,
    speed: float,
    max_chars: int,
    start_after_chunks: int,
    progress_cb=None,
    cache_dir: Path | None = None,
    source_id: str | None = None,
):
    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")

    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, max_chars)
    if not chunks:
        raise ValueError("no chunks to synthesize")

    buffered = []
    started = False
    yield wav_header_float32(KOKORO_SR, channels=1)

    total = len(chunks)
    if progress_cb:
        progress_cb(0, total)
    for i, wav_np in enumerate(
        _generate_kokoro_chunks(
            pipeline=pipeline,
            text_chunks=chunks,
            voice=voice,
            speed=speed,
            cache_dir=cache_dir,
            source_id=source_id,
        ),
        1,
    ):
        if progress_cb:
            progress_cb(i, total)
        chunk_bytes = wav_np.tobytes()
        if not started:
            buffered.append(chunk_bytes)
            if i >= start_after_chunks:
                for b in buffered:
                    yield b
                buffered.clear()
                started = True
        else:
            yield chunk_bytes

    if buffered:
        for b in buffered:
            yield b

    finalize_cached_run(cache_dir, source_id, chunks)
