"""Model loading + preprocessing. Mirrors the two training notebooks exactly."""
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
FACE_MODEL_PATH = Path(os.getenv("FACE_MODEL_PATH", MODEL_DIR / "deepfake_face_detection.keras"))
VOICE_MODEL_PATH = Path(os.getenv("VOICE_MODEL_PATH", MODEL_DIR / "deepfake_voice_model.keras"))

THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # P(fake) >= threshold -> FAKE

# --- video (deepfake-face_detection.ipynb) ---
IMG_SIZE = 224
SEQUENCE_LENGTH = 20
# "notebook": frames /255 and THEN MobileNetV2 preprocess_input (what the notebook trained with).
# "standard": raw 0-255 pixels -> preprocess_input (the usual MobileNetV2 pipeline; use after retraining).
FACE_PREPROCESS = os.getenv("FACE_PREPROCESS", "notebook")

# --- audio (deep-fake-voice-recognition-using-cnn.ipynb) ---
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 500
HOP_LENGTH = 512  # librosa default
MAX_AUDIO_SECONDS = MAX_LEN * HOP_LENGTH / SAMPLE_RATE  # ~16 s analysed

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class ModelStore:
    face = None
    voice = None
    errors: dict = {}


def load_models() -> None:
    import tensorflow as tf  # lazy import keeps `import backend` fast
    import h5py

    for name, path in (("face", FACE_MODEL_PATH), ("voice", VOICE_MODEL_PATH)):
        try:
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            loader = tf.keras.models.load_model
            if name == "face" and h5py.is_hdf5(path):
                import tf_keras

                loader = tf_keras.models.load_model
            setattr(ModelStore, name, loader(path, compile=False))
            ModelStore.errors.pop(name, None)
        except Exception as e:  # keep the API up even if one model is missing
            ModelStore.errors[name] = str(e)


# ---------------------------------------------------------------- video
def extract_frames(video_path: str):
    """Returns (frames[SEQ, 224, 224, 3] in [0,1], number_of_frames_where_a_face_was_found)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read any frames from this video.")

    step = max(1, total // SEQUENCE_LENGTH)
    frames, faces_found, decoded = [], 0, 0

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32))
            continue
        decoded += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        crop = frame
        if len(faces) > 0:
            x, y, w, h = faces[0]
            c = frame[y:y + h, x:x + w]
            if c.size:
                crop = c
                faces_found += 1

        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0)

    cap.release()
    if decoded == 0:
        raise ValueError("Could not decode any frames from this video.")
    return np.array(frames, dtype=np.float32), faces_found


def predict_video(video_path: str) -> dict:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    frames, faces_found = extract_frames(video_path)
    if FACE_PREPROCESS == "standard":
        x = preprocess_input(frames * 255.0)
    else:
        x = preprocess_input(frames.copy())
    p = float(ModelStore.face.predict(x[None], verbose=0)[0][0])
    return {
        **_verdict(p),
        "frames_analysed": SEQUENCE_LENGTH,
        "frames_with_face": faces_found,
    }


# ---------------------------------------------------------------- audio
def extract_mfcc(audio_path: str):
    import librosa

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    if audio.size == 0:
        raise ValueError("Audio file is empty.")
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc[None].astype(np.float32), len(audio) / SAMPLE_RATE


def predict_audio(audio_path: str) -> dict:
    x, duration = extract_mfcc(audio_path)
    p = float(ModelStore.voice.predict(x, verbose=0)[0][0])
    return {
        **_verdict(p),
        "duration_seconds": round(duration, 2),
        "analysed_seconds": round(min(duration, MAX_AUDIO_SECONDS), 2),
    }


# ---------------------------------------------------------------- shared
def _verdict(p_fake: float) -> dict:
    is_fake = p_fake >= THRESHOLD
    return {
        "label": "FAKE" if is_fake else "REAL",
        "fake_probability": round(p_fake, 4),
        "confidence": round(p_fake if is_fake else 1 - p_fake, 4),
        "threshold": THRESHOLD,
    }
