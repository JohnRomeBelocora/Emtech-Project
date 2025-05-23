import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('bird_drone_classifier.h5')

st.title("Bird vs Drone Image Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

target_size = 180  # Should match your training/preprocessing size

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((target_size, target_size))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]
    label = "Drone" if prediction > 0.5 else "Bird"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"Prediction: **{label}** with confidence {confidence:.2f}")
