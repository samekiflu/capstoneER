# app.py
# Author: Same Kiflu
# Description: Streamlit app for Facial Emotion Recognition using a fine-tuned ResNet model
# Automatically handles both single-class and multi-class checkpoints

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


# STREAMLIT CONFIG

st.set_page_config(page_title="Emotion Monitoring System", page_icon="🤖", layout="centered")


# CONFIGURATION
MODEL_PATH = "ResNet_Model_best.pth"   # trained model file
CLASS_NAMES = [
    "angry_elderly_person_face",
    "happy_elderly_person_face",
    "neutral_elderly_person_face",
    "sad_elderly_person_face",
    "surprised_elderly_person_face"
]  # matches your trained dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_classes = len(CLASS_NAMES)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Try loading checkpoint safely
    try:
        model.load_state_dict(checkpoint, strict=True)
        st.success("Model loaded successfully.")
    except RuntimeError as e:
        st.warning(f" Mismatch detected in checkpoint: {e}")
        st.info("Adjusting final layer to match checkpoint...")
        # Remove mismatched fc layer weights and reload safely
        for key in list(checkpoint.keys()):
            if "fc." in key:
                checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=False)
        st.success("Model loaded with adjusted output layer.")

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


# IMAGE TRANSFORM

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# PREDICTION FUNCTION

def predict_emotion(image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)

        # Handle single vs multi-class model dynamically
        if output.shape[1] == 1:
            prob = torch.sigmoid(output).cpu().numpy()[0][0]
            predicted_label = CLASS_NAMES[0]
            return predicted_label, [prob]
        else:
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            predicted_label = CLASS_NAMES[np.argmax(probs)]
            return predicted_label, probs


# STREAMLIT UI

st.title("Emotion Monitoring for Patient Care")
st.write("""
Upload a face image and let the AI model detect emotional states in real time.  
This system can be applied in healthcare settings to help monitor patient well-being,  
track emotional responses, and support caregivers with real-time feedback.
""")

uploaded_file = st.file_uploader("Upload a face image (.jpg or .png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    with col2:
        with st.spinner("Analyzing emotion..."):
            predicted_label, probabilities = predict_emotion(image)

        # Clean up long label names (e.g., "happy_elderly_person_face" → "Happy")
        label = predicted_label.replace("_elderly_person_face", "").capitalize()
        st.subheader(f"Predicted Emotion: **{label}**")

        st.write("### Confidence Levels")
        for emotion, prob in zip(CLASS_NAMES, probabilities):
            st.progress(float(prob))
            st.text(f"{emotion.replace('_elderly_person_face', '').capitalize()}: {prob:.2%}")

        st.success("Emotion detection complete!")

else:
    st.info("Please upload an image to start emotion detection.")

st.caption("Developed by Same Kiflu | Purdue University | Emotion Monitoring System")
