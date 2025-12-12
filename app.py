import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

@st.cache_resource
def load_artifacts():
    # Try multiple filenames in case one failed to upload correctly
    model_files = ['plant_disease_model_subset.keras', 'best_model_subset.h5', 'plant_disease_model_subset.h5']
    model = None
    
    for file in model_files:
        if os.path.exists(file):
            try:
                model = tf.keras.models.load_model(file)
                break
            except Exception:
                continue
                
    if model is None:
        st.error("Model file not found or incompatible. Please check GitHub.")
        
    with open('class_indices.json', 'r') as f:
        class_names = json.load(f)
        
    return model, class_names

model, class_names = load_artifacts()

st.title("🌿 Plant Disease Detector")
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    st.success(f"Prediction: **{class_names[result_index]}**")
    st.info(f"Confidence: {confidence:.2f}%")
