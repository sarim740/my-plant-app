import streamlit as st
import os

# Set environment variable to force legacy Keras behavior before importing TF
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from PIL import Image
import numpy as np
import json

@st.cache_resource
def load_artifacts():
    # Define potential model paths
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        model_path = 'plant_disease_model_subset.keras'
        
    try:
        model = tf.keras.models.load_model(model_path)
        with open('class_indices.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, class_names = load_artifacts()

st.title("🌿 Plant Disease Detector")
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    st.success(f"Prediction: **{class_names[str(result_index)] if isinstance(class_names, dict) else class_names[result_index]}**")
    st.info(f"Confidence: {confidence:.2f}%")
