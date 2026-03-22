import argparse
import asyncio
import tempfile
import uuid
from pathlib import Path
from threading import Lock

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from chatterbox.tts_turbo import ChatterboxTurboTTS

from app.logic import (
    stream_kokoro_wav,
    pack_sentences,
    split_sentences,
    stream_wav,
    synthesize_kokoro_to_wav_path,
    synthesize_to_wav_path,
    stitch_cached_wav_path,
)
from app.ui import ui_html

app = FastAPI()
model_lock = Lock()
MODEL = None
KOKORO_PIPELINE = None
DEVICE = None
VOICE_FILES = {}
PROGRESS = {}
CACHE_DIR = Path("app/cache")
ASSET_VOICES_DIR = Path("app/assets/voices")
DEFAULT_VOICE = ASSET_VOICES_DIR / "voice_clone_me.wav"
KOKORO_DEFAULT_VOICE = "af_heart"
KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
    "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa",
    "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara",
    "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni",
    "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]
progress_lock = Lock()


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    engine: str = "chatterbox"
    voice_path: str | None = None
    exaggeration: float = 0.5
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    max_chars: int = 350
    sleep_every: int = 0
    sleep_seconds: float = 0.0
    speed: float = 0.9
    source_id: str | None = None
    voice_asset: str | None = None


def _cache_source_id(
    base_id: str | None, voice_tag: str | None, engine: str = "chatterbox"
) -> str | None:
    base = base_id or "default"
    base = f"{base}|engine:{engine}"
    if voice_tag and base_id:
        if voice_tag.startswith("asset:"):
            asset_name = voice_tag.split("asset:", 1)[1]
            if asset_name == base_id:
                return base
        return f"{base}|voice:{voice_tag}"
    if voice_tag and not base_id:
        return f"{base}|voice:{voice_tag}"
    return base


def _normalize_engine(engine: str | None) -> str:
    e = (engine or "chatterbox").strip().lower()
    if e not in {"chatterbox", "kokoro"}:
        raise HTTPException(status_code=400, detail="engine must be chatterbox or kokoro")
    return e


def _get_kokoro_pipeline():
    global KOKORO_PIPELINE
    if KOKORO_PIPELINE is not None:
        return KOKORO_PIPELINE
    try:
        from kokoro import KPipeline
        KOKORO_PIPELINE = KPipeline(
            lang_code="a",
            repo_id="hexgrad/Kokoro-82M",
            device=DEVICE,
        )
        return KOKORO_PIPELINE
    except Exception:
        return None


@app.on_event("startup")
def _load_model():
    global MODEL, DEVICE
    if DEVICE is None:
        DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    if MODEL is None:
        MODEL = ChatterboxTurboTTS.from_pretrained(device=DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "kokoro_available": _get_kokoro_pipeline() is not None}


@app.get("/voices")
def list_voices(engine: str = "chatterbox"):
    engine = _normalize_engine(engine)
    if engine == "kokoro":
        return {"voices": KOKORO_VOICES, "default_voice": KOKORO_DEFAULT_VOICE}
    if not ASSET_VOICES_DIR.exists():
        return {"voices": [], "default_voice": None}
    voices = sorted([p.name for p in ASSET_VOICES_DIR.glob("*.wav")])
    default_voice = DEFAULT_VOICE.name if DEFAULT_VOICE.exists() else None
    return {"voices": voices, "default_voice": default_voice}


@app.post("/progress_start")
def progress_start(text: str = Form(...), max_chars: int = Form(350)):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    sentences = split_sentences(text)
    chunks = pack_sentences(sentences, max_chars)
    total = len(chunks)
    job_id = str(uuid.uuid4())
    with progress_lock:
        PROGRESS[job_id] = {"total": total, "done": 0}
    return {"job_id": job_id, "total": total}


@app.get("/progress")
def progress(job_id: str):
    with progress_lock:
        data = PROGRESS.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id not found")
    return data


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest, background_tasks: BackgroundTasks):
    engine = _normalize_engine(req.engine)
    with model_lock:
        try:
            if engine == "kokoro":
                kokoro_pipeline = _get_kokoro_pipeline()
                if kokoro_pipeline is None:
                    raise HTTPException(
                        status_code=503,
                        detail="kokoro engine not available in current environment",
                    )
                if req.voice_path:
                    raise HTTPException(
                        status_code=400,
                        detail="voice_path uploads are only supported for chatterbox",
                    )
                voice_name = (req.voice_asset or KOKORO_DEFAULT_VOICE).strip() or KOKORO_DEFAULT_VOICE
                voice_tag = f"asset:{voice_name}|speed:{req.speed:.3f}"
                out_path = synthesize_kokoro_to_wav_path(
                    pipeline=kokoro_pipeline,
                    text=req.text,
                    voice=voice_name,
                    speed=req.speed,
                    max_chars=req.max_chars,
                    sleep_every=req.sleep_every,
                    sleep_seconds=req.sleep_seconds,
                    cache_dir=CACHE_DIR,
                    source_id=_cache_source_id(req.source_id, voice_tag, engine=engine),
                )
                background_tasks.add_task(out_path.unlink, missing_ok=True)
                return FileResponse(
                    out_path,
                    media_type="audio/wav",
                    filename="speech.wav",
                )

            voice_path = req.voice_path
            voice_tag = None
            if not voice_path and req.voice_asset:
                candidate = ASSET_VOICES_DIR / req.voice_asset
                if not candidate.exists():
                    raise HTTPException(
                        status_code=400, detail="voice_asset not found")
                voice_path = str(candidate)
                voice_tag = f"asset:{req.voice_asset}"
            elif voice_path:
                voice_tag = f"path:{voice_path}"
            elif DEFAULT_VOICE.exists():
                voice_path = str(DEFAULT_VOICE)
                voice_tag = "asset:voice_clone_me.wav"
            out_path = synthesize_to_wav_path(
                model=MODEL,
                device=DEVICE,
                text=req.text,
                voice_path=voice_path,
                exaggeration=req.exaggeration,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_chars=req.max_chars,
                sleep_every=req.sleep_every,
                sleep_seconds=req.sleep_seconds,
                speed=req.speed,
                cache_dir=CACHE_DIR,
                source_id=_cache_source_id(req.source_id, voice_tag, engine=engine),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    background_tasks.add_task(out_path.unlink, missing_ok=True)
    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="speech.wav",
    )


@app.post("/upload_voice")
def upload_voice(file: UploadFile = File(...)):
    suffix = Path(file.filename or "voice.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(
        prefix="chatterbox_voice_", suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with tmp_path.open("wb") as f:
        f.write(file.file.read())
    voice_id = str(uuid.uuid4())
    VOICE_FILES[voice_id] = tmp_path
    return {"voice_id": voice_id}


@app.get("/ui")
def ui():
    return HTMLResponse(ui_html())


@app.get("/synthesize_cached")
def synthesize_cached(
    background_tasks: BackgroundTasks,
    source_id: str | None = None,
    voice_id: str | None = None,
    voice_asset: str | None = None,
    engine: str = "chatterbox",
    speed: float = 0.9,
):
    engine = _normalize_engine(engine)
    base_id = source_id if source_id else None
    voice_tag = None
    if engine == "kokoro":
        voice_name = (voice_asset or KOKORO_DEFAULT_VOICE).strip() or KOKORO_DEFAULT_VOICE
        voice_tag = f"asset:{voice_name}|speed:{speed:.3f}"
    elif voice_id:
        voice_tag = f"upload:{voice_id}"
    elif voice_asset:
        voice_tag = f"asset:{voice_asset}"
    cache_id = _cache_source_id(base_id, voice_tag, engine=engine)
    if not cache_id:
        raise HTTPException(status_code=400, detail="source_id is required")
    try:
        out_path = stitch_cached_wav_path(CACHE_DIR, cache_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    background_tasks.add_task(out_path.unlink, missing_ok=True)
    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="speech.wav",
    )


@app.get("/synthesize_stream")
def synthesize_stream(
    text: str,
    engine: str = "chatterbox",
    voice_id: str | None = None,
    voice_asset: str | None = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_chars: int = 350,
    start_after_chunks: int = 3,
    job_id: str | None = None,
    speed: float = 0.9,
    source_id: str | None = None,
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    engine = _normalize_engine(engine)
    voice_path = None
    voice_tag = None
    if engine == "kokoro":
        kokoro_pipeline = _get_kokoro_pipeline()
        if kokoro_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="kokoro engine not available in current environment",
            )
        if voice_id:
            raise HTTPException(
                status_code=400,
                detail="voice uploads are only supported for chatterbox",
            )
        voice_name = (voice_asset or KOKORO_DEFAULT_VOICE).strip() or KOKORO_DEFAULT_VOICE
        voice_tag = f"asset:{voice_name}|speed:{speed:.3f}"
    elif voice_id:
        voice_path = VOICE_FILES.get(voice_id)
        if voice_path is None or not voice_path.exists():
            raise HTTPException(status_code=400, detail="invalid voice_id")
        voice_tag = f"upload:{voice_id}"
    elif voice_asset:
        candidate = ASSET_VOICES_DIR / voice_asset
        if not candidate.exists():
            raise HTTPException(
                status_code=400, detail="voice_asset not found")
        voice_path = candidate
        voice_tag = f"asset:{voice_asset}"
    elif DEFAULT_VOICE.exists():
        voice_path = DEFAULT_VOICE
        voice_tag = "asset:voice_clone_me.wav"

    async def _gen():
        done = 0
        total = 0
        try:
            if engine == "chatterbox" and voice_path:
                with model_lock:
                    MODEL.prepare_conditionals(
                        str(voice_path),
                        exaggeration=exaggeration)

            def _progress_cb(d, t):
                nonlocal done, total
                done = d
                total = t
                if not job_id:
                    return
                with progress_lock:
                    if job_id in PROGRESS:
                        PROGRESS[job_id]["done"] = d
                        PROGRESS[job_id]["total"] = t
                    else:
                        PROGRESS[job_id] = {"done": d, "total": t}

            if engine == "kokoro":
                iterator = stream_kokoro_wav(
                    pipeline=kokoro_pipeline,
                    text=text,
                    voice=voice_name,
                    speed=speed,
                    max_chars=max_chars,
                    start_after_chunks=max(1, start_after_chunks),
                    progress_cb=_progress_cb,
                    cache_dir=CACHE_DIR,
                    source_id=_cache_source_id(source_id, voice_tag, engine=engine),
                )
            else:
                iterator = stream_wav(
                    model=MODEL,
                    device=DEVICE,
                    text=text,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_chars=max_chars,
                    start_after_chunks=max(1, start_after_chunks),
                    progress_cb=_progress_cb,
                    speed=speed,
                    cache_dir=CACHE_DIR,
                    source_id=_cache_source_id(source_id, voice_tag, engine=engine),
                )
            while True:
                with model_lock:
                    try:
                        b = next(iterator)
                    except StopIteration:
                        break
                yield b
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected; stop immediately.
            pass
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        finally:
            if job_id:
                with progress_lock:
                    if job_id in PROGRESS:
                        PROGRESS[job_id]["done"] = done or PROGRESS[job_id][
                            "done"]
                        PROGRESS[job_id]["total"] = total or PROGRESS[job_id]["total"]
            if engine == "chatterbox" and voice_id and voice_path and voice_path.exists():
                voice_path.unlink(missing_ok=True)
                VOICE_FILES.pop(voice_id, None)

    return StreamingResponse(_gen(), media_type="audio/wav")


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", type=str, default=None, help="mps or cpu")
    args = ap.parse_args()

    if args.device is None:
        DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        DEVICE = args.device

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
