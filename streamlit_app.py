import logging
import sys

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Diabetes Detection AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: #1e293b;
        padding: clamp(10px, 3vw, 20px);
    }
    .stApp {
        background: #1e293b;
        max-width: 95vw;
        margin: 0 auto;
    }

    @media (min-width: 768px) {
        .stApp {
            max-width: 85vw;
        }
    }

    @media (min-width: 1024px) {
        .stApp {
            max-width: 1000px;
        }
    }

    @media (min-width: 1440px) {
        .stApp {
            max-width: 1200px;
        }
    }

    div[data-testid="stFileUploader"] {
        background: #334155;
        border-radius: 12px;
        padding: 20px;
        border: none;
    }
    div[data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stFileUploader"] section {
        border: 2px dashed #475569 !important;
        border-radius: 8px !important;
        background: #1e293b !important;
    }
    div[data-testid="stFileUploader"] section button {
        background: #475569 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-size: 0.9rem !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 24px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
        transform: translateY(-1px);
    }

    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #475569;
        background: #334155;
        color: #e2e8f0;
        padding: 12px 16px;
        font-size: 0.95rem;
    }
    .stTextInput input::placeholder {
        color: #94a3b8;
    }
    .stTextInput input:focus {
        border-color: #ef4444;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.1);
    }
    .stTextInput label {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-bottom: 6px !important;
    }

    h1 {
        color: white;
        text-align: center;
        font-size: clamp(1.5rem, 5vw, 2rem);
        font-weight: 700;
        margin: clamp(15px, 3vh, 30px) 0 clamp(10px, 2vh, 20px) 0;
    }

    .main-container {
        background: #2d3748;
        border-radius: 32px;
        padding: clamp(20px, 5vw, 30px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    .login-container {
        background: #2d3748;
        border-radius: 16px;
        padding: clamp(25px, 6vw, 35px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        margin-top: clamp(20px, 5vh, 40px);
    }

    .image-preview {
        background: #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 0;
        text-align: center;
    }

    .image-preview img {
        max-height: 25vh;
        width: 100%;
        object-fit: contain;
    }

    @media (min-width: 768px) {
        .image-preview img {
            max-height: 28vh;
        }
    }

    @media (min-width: 1024px) {
        .image-preview img {
            max-height: 32vh;
        }
    }

    .result-box {
        background: #334155;
        border-radius: 12px;
        padding: clamp(15px, 4vw, 25px);
        margin: clamp(10px, 2vh, 20px) 0;
        text-align: center;
        border-left: 4px solid;
    }

    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
    }

    .stTextInput label p {
        font-size: clamp(0.85rem, 2.5vw, 0.9rem) !important;
    }

    .stButton {
        margin-top: clamp(10px, 2vh, 15px);
    }

    [data-testid="column"] {
        vertical-align: top;
    }

    div[data-testid="stFileUploader"],
    .image-preview {
        margin-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def check_login(username, password):
    return username == os.getenv("LOGIN_USER") and password == os.getenv("LOGIN_PASS")

if not st.session_state.authenticated:
    st.markdown("<h1>🩺 Diabetes Detection AI</h1>", unsafe_allow_html=True)
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

st.markdown("<h1>🩺 Diabetes Detection AI</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_vit():
    try:
        processor = ViTImageProcessor.from_pretrained(".")
        model = ViTForImageClassification.from_pretrained(".")
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="uploader")

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")

        with col2:
            # st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            # st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-bottom: 10px;'>Uploaded Image</p>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)

        labels = {
            0: "No DR",
            1: "Mild DR",
            2: "Moderate DR",
            3: "Severe DR",
            4: "Proliferative DR"
        }

        label_colors = {
            0: "#10b981",
            1: "#3b82f6",
            2: "#f59e0b",
            3: "#f97316",
            4: "#ef4444"
        }

        processor, model = load_vit()
        if processor is None or model is None:
            st.error("Model not loaded. Cannot make prediction.")
        else:
            try:
                with st.spinner("Analyzing image..."):
                    inputs = processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        output = model(**inputs)
                        class_idx = torch.argmax(output.logits, dim=1).item()

                result_color = label_colors[class_idx]
                st.markdown(f"""
                    <div class="result-box" style='border-left-color: {result_color};'>
                    <p style='color: #94a3b8; font-size: 0.85rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>Diagnosis</p>
                    <h2 style='color: {result_color}; font-size: 1.8rem; margin: 0; font-weight: 700;'>{labels[class_idx]}</h2>
                    </div>
                """, unsafe_allow_html=True)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    with open("predictions.txt", "a") as f:
                        f.write(f"{timestamp} | {uploaded_file.name} | {labels[class_idx]}\n")
                except Exception as e:
                    st.warning(f"Prediction successful but failed to log: {str(e)}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")