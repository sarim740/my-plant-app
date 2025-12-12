import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🌿 Plant Disease Classifier")

# Load the H5 model
model = tf.keras.models.load_model("model.h5", compile=False)

# Replace with your actual class labels
class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    # Add all other classes...
]

def preprocess(img):
    img = img.resize((224, 224))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    processed_img = preprocess(img)
    predictions = model.predict(processed_img)
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Prediction: **{predicted_class}**")
