import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import pickle

# Load the trained MobileNet model with caching
@st.cache_resource(show_spinner=False)
def load_trained_model():
    try:
        model = load_model("best_mobilenet_model.keras")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load label encoder
@st.cache_resource()
def load_label_encoder():
    try:
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        st.error(f"❌ Error loading label encoder: {e}")
        return None

# Preprocess image for model
def preprocess_image(image):
    """Prepares an image for model inference without altering the original display image."""
    try:
        image_resized = cv2.resize(image, (128, 128))  # Resize to match model input shape
        image_resized = image_resized.astype("float32") / 255.0  # Normalize pixel values
        image_resized = np.expand_dims(image_resized, axis=0)  # Expand dimensions for batch
        return image_resized
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# Classify image
def classify_image(model, label_encoder, image):
    processed_image = preprocess_image(image)
    if processed_image is None:
        return None, None
    try:
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
        confidence = np.max(predictions) * 100
        return predicted_class, confidence
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
        return None, None

# Streamlit UI
st.title("🌿 Plant Deficiency Classification using MobileNet")
uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Keep original BGR format
    image_for_processing = original_image.copy()  # Copy for processing

    st.image(original_image, caption="📷 Uploaded Image (Original Colors)", use_column_width=True, channels="BGR")

    model = load_trained_model()
    label_encoder = load_label_encoder()

    if model and label_encoder:
        predicted_class, confidence = classify_image(model, label_encoder, image_for_processing)
        if predicted_class:
            st.success(f"✅ Prediction: **{predicted_class}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")
