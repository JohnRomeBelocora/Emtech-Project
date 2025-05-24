import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Conditional imports for TensorFlow 2.10
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
except:
    from keras.models import load_model
    from keras.preprocessing import image

@st.cache_resource
def load_my_model():
    model_path = "bird_drone_classifier.h5"
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=1WgF1JCIAUlq65eSntmS0kCvdRvWRLrhH"
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_my_model()

st.title("Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an image...", type="jpeg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image
    img = img.resize((150, 150))  # Match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    try:
        prediction = model.predict(img_array)
        class_names = ['Bird', 'Drone']
        predicted_class = class_names[int(np.round(prediction[0][0]))]
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {np.max(prediction)*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
