import os
import io
import time
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import librosa
import librosa.display
import tensorflow as tf
import joblib


st.set_page_config(page_title="Animal Voice Classifier (ANN)", page_icon="🐾", layout="wide")

MODEL_PATH = "animal_voice_ann.keras"
SCALER_PATH = "scaler.joblib"
ENCODER_PATH = "label_encoder.joblib"


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


def extract_features_from_path(fp, target_sr=22050, n_mfcc=40):
    y, sr = librosa.load(fp, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError("Audio kosong atau gagal dibaca")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate(
        [
            mfcc.mean(axis=1), mfcc.std(axis=1),
            delta.mean(axis=1), delta.std(axis=1),
            delta2.mean(axis=1), delta2.std(axis=1)
        ],
        axis=0
    )

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat, y, sr


def predict_from_tempfile(model, scaler, encoder, fp):
    feat, y, sr = extract_features_from_path(fp)
    x = scaler.transform([feat])
    prob = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(prob))
    label = encoder.inverse_transform([idx])[0]
    conf = float(prob[idx])
    proba_df = pd.DataFrame({"label": encoder.classes_, "probability": prob}).sort_values("probability", ascending=False)
    return label, conf, proba_df, y, sr


def plot_waveform(y, sr, title):
    fig = plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_melspec(y, sr, title):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure()
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    return fig


def ensure_model_files():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH] if not os.path.exists(p)]
    return missing


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .stMetric { background: rgba(255,255,255,0.04); padding: 12px; border-radius: 16px; }
    .card { background: rgba(255,255,255,0.04); padding: 18px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); }
    .small { opacity: 0.75; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐾 Animal Voice Classifier (ANN + MFCC)")
st.write("Audio → MFCC (mean/std + delta + delta²) → ANN → Probabilitas kelas")

missing = ensure_model_files()
if missing:
    st.error("File model belum ada di folder app. Taruh file berikut di sebelah app.py:")
    st.write(missing)
    st.stop()

model, scaler, encoder = load_assets()

with st.sidebar:
    st.header("Kontrol")
    st.caption("Pastikan audio jelas, noise rendah, durasi 1–5 detik.")
    st.divider()
    st.subheader("Kelas yang dikenal model")
    st.write(list(encoder.classes_))
    st.divider()
    topk = st.slider("Top-K Probabilitas", 3, min(15, len(encoder.classes_)), 5)
    show_plots = st.toggle("Tampilkan Waveform & Spectrogram", value=True)
    st.divider()
    st.caption("Jika MP3 bermasalah, coba WAV. Jika perlu, install FFmpeg.")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Upload audio")
up = st.file_uploader("Pilih file audio", type=["wav", "mp3", "flac", "ogg", "aiff", "aif", "m4a"])
st.markdown("</div>", unsafe_allow_html=True)

    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{up.name}") as tmp:
            tmp.write(up.getbuffer())
            tmp_path = tmp.name

        try:
            label, conf, proba_df, y, sr = predict_from_tempfile(model, scaler, encoder, tmp_path)

            c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
            with c1:
                st.metric("Prediksi", str(label))
            with c2:
                st.metric("Confidence", f"{conf:.3f}")
            with c3:
                st.markdown('<div class="small">', unsafe_allow_html=True)
                st.write(f"Sample rate: {sr} Hz")
                st.write(f"Durasi: {len(y)/sr:.2f} detik")
                st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Top Probabilitas")
            st.dataframe(proba_df.head(topk), use_container_width=True, hide_index=True)

            figp = plt.figure()
            top_df = proba_df.head(topk).iloc[::-1]
            plt.barh(top_df["label"], top_df["probability"])
            plt.title("Top-K Probability")
            plt.tight_layout()
            st.pyplot(figp)

            if show_plots:
                st.subheader("Visualisasi audio")
                st.pyplot(plot_waveform(y, sr, f"Waveform | {label} ({conf:.3f})"))
                st.pyplot(plot_melspec(y, sr, f"Mel-Spectrogram | {label} ({conf:.3f})"))

            st.audio(up.getvalue())

        except Exception as e:
            st.error(f"Gagal memproses audio: {e}")

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

st.divider()
st.caption("Catatan: Model ANN ini bergantung pada kualitas label. Kalau label dataset kamu salah (misal cuma 1 kelas), semua hasil prediksi akan terlihat 'meyakinkan' tapi sebenarnya tidak valid.")
