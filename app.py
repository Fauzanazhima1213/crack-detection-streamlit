import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="AI Deteksi Keretakan",
    page_icon="🧱",
    layout="centered"
)

st.title("🧱 AI Deteksi Keretakan")
st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan gambar apakah terdapat "
    "keretakan atau tidak menggunakan model CNN berformat H5."
)

# =========================
# KONFIGURASI MODEL
# =========================
MODEL_PATH = "crack_detection_model.h5"

# ID Google Drive dari file model .h5 kamu
FILE_ID = "1LFNc3zllABh4jaWI4imv15l9FIgu7WTD"

IMG_SIZE = (128, 128)


# =========================
# DOWNLOAD DAN LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("Mengunduh model dari Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


try:
    model = load_model()
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error("Model gagal dimuat. Pastikan file Google Drive sudah dibuat public.")
    st.write(e)
    st.stop()


# =========================
# FUNGSI PREPROCESS GAMBAR
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar untuk dideteksi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Gambar yang Diupload")
    st.image(image, use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    # Saat training:
    # Negative = 0 = Tidak Retak
    # Positive = 1 = Retak
    crack_probability = float(prediction)
    no_crack_probability = 1 - crack_probability

    st.subheader("Hasil Prediksi")

    if crack_probability >= 0.5:
        st.error("HASIL: TERDETEKSI RETAK")
        st.write(f"Tingkat keyakinan retak: **{crack_probability * 100:.2f}%**")
    else:
        st.success("HASIL: TIDAK TERDETEKSI RETAK")
        st.write(f"Tingkat keyakinan tidak retak: **{no_crack_probability * 100:.2f}%**")

    st.write("Probabilitas Retak:")
    st.progress(crack_probability)

    st.caption(
        "Catatan: model ini merupakan model klasifikasi, sehingga hanya menentukan "
        "Retak atau Tidak Retak, bukan menandai posisi retakan pada gambar."
    )

else:
    st.info("Silakan upload gambar terlebih dahulu.")
