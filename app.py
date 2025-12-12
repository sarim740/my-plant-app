import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

@st.cache_resource
def load_model_and_labels():
    # Identify the model file
    model_path = 'plant_disease_model_subset.keras' 
    if not os.path.exists(model_path):
        model_path = 'model.h5'
        
    # FIX: We use compile=False to avoid layer-specific argument errors 
    # during the initial loading phase.
    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open('class_indices.json', 'r') as f:
        class_names = json.load(f)
    
    return model, class_names

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

try:
    model, class_names = load_model_and_labels()
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Tip: Try saving your model in Colab as 'model.h5' and re-uploading.")
    model = None

st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to identify potential diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
    
    # Handle both list and dict formats for class_indices
    if isinstance(class_names, dict):
        result_text = class_names.get(str(result_index), class_names.get(result_index, "Unknown"))
    else:
        result_text = class_names[result_index]

    st.write(f"### Prediction: **{result_text}**")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")
