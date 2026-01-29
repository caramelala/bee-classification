import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

MODEL_PATH = "model_baseline.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🐝 Klasifikasi Lebah Tanpa Sengat")

uploaded_file = st.file_uploader(
    "Upload gambar lebah",
    type=["jpg","jpeg","png"]
)

CLASS_NAMES = ["kelas1","kelas2","kelas3","kelas4"]  # ganti sesuai kelasmu

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Gambar input", width=300)

    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    idx = np.argmax(pred)

    st.success(f"Hasil prediksi: {CLASS_NAMES[idx]}")

