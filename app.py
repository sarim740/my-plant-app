
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_resource
def load_labels():
    with open('class_indices.json', 'r') as f:
        return json.load(f)

model = load_model()
class_names = load_labels()

st.title("🌿 Plant Disease Detector")
file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])

if file:
    img = Image.open(file).resize((224, 224))
    st.image(img, use_container_width=True)
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    prediction = model.predict(x)
    idx = np.argmax(prediction)
    st.success(f"Result: {class_names[idx]} ({np.max(prediction)*100:.1f}%)")
    