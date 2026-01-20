import streamlit as st
import tempfile
import torch

from modalities.image.infer import analyze_image
from modalities.audio.infer import analyze_audio
from modalities.video.infer import analyze_video
from modalities.text.infer import analyze_text
from core.consistency import verdict_from_scores

st.set_page_config(
    page_title="OmniForensics-Stream",
    layout="centered"
)

st.title("🕵️ OmniForensics-Stream")
st.caption("Multimodal AI-generated content forensic system")

uploaded = st.file_uploader(
    "Upload image / audio / video / text",
    type=["png", "jpg", "jpeg", "wav", "mp3", "mp4", "txt"]
)

if uploaded:
    suffix = uploaded.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as f:
        f.write(uploaded.read())
        path = f.name

    image_score = None
    audio_score = None
    text_score = None

    st.subheader("🔍 Analysis")

    # ---------------- IMAGE ----------------
    if suffix in ["png", "jpg", "jpeg"]:
        image_score = analyze_image(path)
        st.metric("Image Authenticity Score", f"{1 - image_score:.2f}")
        st.metric("Image Fake Probability", f"{image_score:.2f}")

    # ---------------- AUDIO ----------------
    elif suffix in ["wav", "mp3"]:
        audio_score = analyze_audio(path)
        st.metric("Audio Authenticity Score", f"{1 - audio_score:.2f}")
        st.metric("Audio Fake Probability", f"{audio_score:.2f}")

    # ---------------- VIDEO ----------------
    elif suffix in ["mp4"]:
        result = analyze_video(path)
        image_score = result["score"]
        st.metric("Video Fake Probability", f"{image_score:.2f}")
        st.caption(f"Frames analyzed: {result['frames']}")

    # ---------------- TEXT ----------------
    elif suffix in ["txt"]:
        text = uploaded.read().decode("utf-8")
        text_score = analyze_text(text)
        st.metric("Text Fake Probability", f"{text_score:.2f}")

    # ---------------- CONSISTENCY ----------------
    if image_score is not None or audio_score is not None:
        verdict = verdict_from_scores(
            image_score=image_score or 0.0,
            audio_score=audio_score or 0.0,
            text_score=text_score
        )

        st.divider()
        st.subheader("🧠 Final Verdict")

        st.write(f"**Verdict:** {verdict['verdict']}")
        st.write(f"**Confidence:** {verdict['confidence']:.2f}")
        st.write(f"**Reason:** {verdict['reason']}")



