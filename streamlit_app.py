import streamlit as st
import os
import uuid
from PIL import Image
import numpy as np
from model_utils import BreastCancerClassifier

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Breast Cancer Classification App")
st.markdown(
    "Upload a histopathological image to classify it as **Benign** or **Malignant**."
)

# Initialize classifier (cached so it only loads once)
@st.cache_resource
def load_classifier():
    model_path = "models/breast_cancer_model.keras"
    return BreastCancerClassifier(model_path if os.path.exists(model_path) else None)

classifier = load_classifier()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            # Save to a temporary file with a safe random name
            tmp_path = f"/tmp/{uuid.uuid4().hex}.png"
            try:
                image.save(tmp_path)
                result = classifier.predict(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if result:
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            risk_level = result["risk_level"]
            interpretation = result["interpretation"]

            if predicted_class == "Malignant":
                st.error(f"### ⚠️ Prediction: {predicted_class}")
            else:
                st.success(f"### ✅ Prediction: {predicted_class}")

            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{confidence:.1%}")
            col2.metric("Risk Level", risk_level)

            st.markdown(f"**Benign probability:** {result['benign_probability']:.1%}")
            st.markdown(f"**Malignant probability:** {result['malignant_probability']:.1%}")
            st.info(f"**Interpretation:** {interpretation}")

            st.warning(
                "⚕️ **Disclaimer:** This tool is for research/educational purposes only "
                "and is not a substitute for professional medical advice."
            )
        else:
            st.error("Could not process the image. Please try a different file.")

st.markdown("---")
st.markdown(
    "**About:** This app uses a ResNet50-based deep learning model trained on "
    "histopathological breast cancer images to distinguish benign from malignant tissue."
)
