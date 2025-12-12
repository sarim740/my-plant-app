import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

@st.cache_resource
def load_artifacts():
    # Attempt to load using standard method
    try:
        model = tf.keras.models.load_model('plant_disease_model_subset.keras')
    except Exception:
        # If standard load fails, try legacy H5 loading
        model = tf.keras.models.load_model('best_model_subset.h5')
    
    with open('class_indices.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_artifacts()

st.title("🌿 Plant Disease Detector")
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
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
