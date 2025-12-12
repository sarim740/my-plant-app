import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

@st.cache_resource
def load_model_and_labels():
    # Force the use of the .h5 or .keras file present in your repo
    model_path = 'plant_disease_model_subset.keras' 
    if not os.path.exists(model_path):
        model_path = 'model.h5'
        
    # Load model using the TF-integrated Keras
    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open('class_indices.json', 'r') as f:
        class_names = json.load(f)
    
    return model, class_names

try:
    model, class_names = load_model_and_labels()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Initialization Error: {e}")
    model = None

st.title("🌿 Plant Disease Detector")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    
    st.write(f"### Prediction: {class_names[str(result_index)]}")
    st.write(f"**Confidence:** {np.max(predictions)*100:.2f}%")
