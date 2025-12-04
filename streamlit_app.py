import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from datetime import datetime

st.title("Diabetic Retinopathy Classification")

@st.cache_resource
def load_vit():
    processor = ViTImageProcessor.from_pretrained(".")
    model = ViTForImageClassification.from_pretrained(".")
    model.eval()
    return processor, model

uploaded_file = st.file_uploader("Drop retinal image here", type=["png", "jpg", "jpeg"], key="uploader")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", width='stretch')
    with col2:
        img_resized = img.resize((224, 224))
        st.image(img_resized, caption="Preprocessed (224x224)", width='stretch')

    labels = {
        0: "0 - No DR",
        1: "1 - Mild",
        2: "2 - Moderate",
        3: "3 - Severe",
        4: "4 - Proliferative DR"
    }

    processor, model = load_vit()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        class_idx = torch.argmax(output.logits, dim=1).item()

    st.success(f"Prediction: {labels[class_idx]}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("predictions.txt", "a") as f:
        f.write(f"{timestamp} | {uploaded_file.name} | {labels[class_idx]}\n")