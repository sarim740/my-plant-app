import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

@st.cache_resource
def load_artifacts():
    # Load the universal H5 file
    model = tf.keras.models.load_model('model.h5', compile=False)
    with open('class_indices.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

st.title("🌿 Plant Disease Detector")

try:
    model, class_names = load_artifacts()
    
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        preds = model.predict(img_array)
        idx = np.argmax(preds)
        
        # Map result
        result = class_names[str(idx)] if isinstance(class_names, dict) else class_names[idx]
        st.success(f"Prediction: {result} ({np.max(preds)*100:.2f}%)")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
