import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load model
model_path = 'plant_disease_model_subset.h5' # Ensure this path is correct
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

model = load_model(model_path)

# Class labels - These should match the labels used during training
# Assuming `class_labels` was defined in a previous cell, I'll hardcode it here based on common PlantVillage dataset labels.
# In a real scenario, this would be loaded or passed from the training script.
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

img_size = 224 # Assuming image size used for training

st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale to [0,1] as done in training

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    st.success(f"Prediction: {predicted_class_label}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Optional: Display top N predictions
    st.subheader("Top 3 Predictions:")
    top_indices = np.argsort(predictions[0])[::-1][:3]
    for i in top_indices:
        st.write(f"- {class_labels[i]}: {predictions[0][i]*100:.2f}%")
