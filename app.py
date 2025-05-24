import streamlit as st
from PIL import Image
import numpy as np
import os

# Force TensorFlow to use CPU only (reduces cloud errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

@st.cache_resource
def load_my_model():
    model_path = "bird_drone_classifier.h5"
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=1WgF1JCIAUlq65eSntmS0kCvdRvWRLrhH"
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"❌ Model download failed: {str(e)}")
            return None
    
    try:
        # Critical fix: Use legacy Keras loading
        from tensorflow.python.keras.models import load_model
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

model = load_my_model()
if model is None:
    st.stop()  # Stop if model fails to load

st.title("Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)
        
        # Get model's expected input shape dynamically
        target_size = model.input_shape[1:3]  # (height, width)
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        class_names = ['Bird', 'Drone']
        predicted_class = class_names[int(np.round(prediction[0][0]))]
        
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {np.max(prediction)*100:.2f}%")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
