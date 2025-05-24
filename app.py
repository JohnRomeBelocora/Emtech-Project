import streamlit as st
from PIL import Image
import numpy as np
import os

# Improved import handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
except ImportError:
    from keras.models import load_model
    from keras.preprocessing import image

@st.cache_resource
def load_my_model():
    model_path = "bird_drone_classifier.h5"
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=1WgF1JCIAUlq65eSntmS0kCvdRvWRLrhH"
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Model download failed: {str(e)}")
            return None
    
    try:
        # Critical change: Add compile=False and custom_objects if needed
        model = load_model(model_path, compile=False)
        st.write(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_my_model()
if model is None:
    st.stop()  # Stop execution if model fails to load

st.title("Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')  # Ensure RGB format
        st.image(img, caption='Uploaded Image', use_container_width=True)
        
        # Verify resize matches model's expected input
        target_size = model.input_shape[1:3]  # Get (height, width) from model
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0,1]
        
        # Make prediction
        prediction = model.predict(img_array)
        class_names = ['Bird', 'Drone']
        predicted_class = class_names[int(np.round(prediction[0][0]))]
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {np.max(prediction)*100:.2f}%")
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
