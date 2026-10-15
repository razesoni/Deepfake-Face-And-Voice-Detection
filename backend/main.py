import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import inference as inf

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    inf.load_models()
    yield


app = FastAPI(title="Deepfake Detector API", version="1.0.0", lifespan=lifespan)


def _save_upload(file: UploadFile, allowed: set) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _require(model, name: str):
    if model is None:
        raise HTTPException(503, f"{name} model not loaded: {inf.ModelStore.errors.get(name, 'unknown error')}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"video": inf.ModelStore.face is not None, "audio": inf.ModelStore.voice is not None},
        "errors": inf.ModelStore.errors,
        "threshold": inf.THRESHOLD,
    }


@app.post("/predict/video")
def predict_video(file: UploadFile = File(...)):
    _require(inf.ModelStore.face, "face")
    path = _save_upload(file, VIDEO_EXT)
    try:
        return inf.predict_video(path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(path)


@app.post("/predict/audio")
def predict_audio(file: UploadFile = File(...)):
    _require(inf.ModelStore.voice, "voice")
    path = _save_upload(file, AUDIO_EXT)
    try:
        return inf.predict_audio(path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, f"Could not process audio: {e}")
    finally:
        os.unlink(path)
