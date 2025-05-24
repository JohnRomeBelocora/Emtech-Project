import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your model
@st.cache_resource
def load_my_model():
    return load_model("bird_drone_classifier.h5")

model = load_my_model()

st.title("Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image
    img = img.resize((150, 150))  # Adjust if your model expects different dimensions
    
    # Convert image to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    
    # Check the shape before prediction
    st.write(f"Input shape: {img_array.shape}")
    
    try:
        # Make prediction
        prediction = model.predict(img_array)
        
        # Assuming binary classification (0: bird, 1: drone)
        class_names = ['Bird', 'Drone']
        predicted_class = class_names[int(np.round(prediction[0][0]))]
        
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {np.max(prediction)*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")