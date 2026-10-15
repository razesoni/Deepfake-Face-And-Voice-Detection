import os

import requests
import streamlit as st

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="centered")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000")).rstrip("/")

try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    st.sidebar.success("API online")
    st.sidebar.write(f"Video model: {'✅' if health['models']['video'] else '❌'}")
    st.sidebar.write(f"Audio model: {'✅' if health['models']['audio'] else '❌'}")
    for name, err in health.get("errors", {}).items():
        st.sidebar.caption(f"{name}: {err}")
except Exception:
    st.sidebar.error("API offline")
    st.sidebar.caption("Start it with: uvicorn backend.main:app --port 8000")


# ---------------------------------------------------------------- helpers
def call_api(endpoint: str, uploaded):
    try:
        r = requests.post(
            f"{api_url}{endpoint}",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
            timeout=600,
        )
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        return None
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
        return None
    return r.json()


def show_result(res: dict):
    fake = res["label"] == "FAKE"
    (st.error if fake else st.success)(f"**{'⚠️ Likely FAKE' if fake else '✅ Likely REAL'}**")
    c1, c2 = st.columns(2)
    c1.metric("Fake probability", f"{res['fake_probability']:.1%}")
    c2.metric("Confidence", f"{res['confidence']:.1%}")
    st.progress(res["fake_probability"], text=f"Fake probability (threshold {res['threshold']:.0%})")


# ---------------------------------------------------------------- UI
st.title("🛡️ Deepfake Detector")
st.caption("Upload a video or an audio clip. Results are probabilistic, not proof.")

tab_video, tab_audio = st.tabs(["🎬 Video (face)", "🎙️ Audio (voice)"])

with tab_video:
    video = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid")
    if video:
        st.video(video)
        if st.button("Analyse video", type="primary"):
            with st.spinner("Extracting faces and running the CNN-LSTM…"):
                res = call_api("/predict/video", video)
            if res:
                show_result(res)
                st.caption(f"Face found in {res['frames_with_face']} of {res['frames_analysed']} sampled frames "
                           "(the rest fall back to the full frame).")
                if res["frames_with_face"] < res["frames_analysed"] // 2:
                    st.warning("Few faces were detected, so this result is less reliable.")

with tab_audio:
    audio = st.file_uploader("Audio file", type=["wav", "mp3", "flac", "ogg", "m4a"], key="aud")
    if audio:
        st.audio(audio)
        if st.button("Analyse audio", type="primary"):
            with st.spinner("Extracting MFCCs and running the CNN…"):
                res = call_api("/predict/audio", audio)
            if res:
                show_result(res)
                st.caption(f"Analysed the first {res['analysed_seconds']}s of {res['duration_seconds']}s.")
