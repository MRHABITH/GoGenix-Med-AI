import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageFilter
from skimage.filters import threshold_otsu
from skimage import exposure
import os

# -------------------------------
# Load Models Once
# -------------------------------
@st.cache_resource
def load_models():
    brain_model_path = "models/brain_tumor_model.keras"
    chest_model_path = "models/chest_xray_model.keras"

    if os.path.exists(brain_model_path):
        brain_model = load_model(brain_model_path)
    else:
        from models.build_models import build_brain_tumor_model
        brain_model = build_brain_tumor_model()

    if os.path.exists(chest_model_path):
        chest_model = load_model(chest_model_path)
    else:
        from models.build_models import build_chest_xray_model
        chest_model = build_chest_xray_model()

    return brain_model, chest_model

brain_model, chest_model = load_models()

# -------------------------------
# Image Type Detection
# -------------------------------
def detect_image_type(image: Image.Image) -> str:
    gray = image.convert("L")
    arr = np.array(gray)

    # -----------------------------
    # FEATURE 1: Average brightness
    # -----------------------------
    avg_intensity = np.mean(arr)

    # -----------------------------
    # FEATURE 2: Contrast
    # -----------------------------
    contrast = np.std(arr)

    # -----------------------------
    # FEATURE 3: Edge Strength
    # -----------------------------
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_strength = np.mean(np.array(edges))

    # -----------------------------
    # FEATURE 4: Entropy (texture)
    # X-rays = high entropy (bones)
    # MRI = smoother image
    # -----------------------------
    hist = exposure.histogram(arr)[0]
    hist = hist / np.sum(hist)
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])

    # -----------------------------
    # FEATURE 5: Bright bone area
    # X-rays have large bright regions (white bones)
    # -----------------------------
    try:
        otsu = threshold_otsu(arr)
    except:
        otsu = 140

    bright_pixels = np.sum(arr > otsu)
    bright_ratio = bright_pixels / arr.size

    # -----------------------------
    # Weighted scoring system
    # -----------------------------
    score = 0

    # X-ray indicators
    if avg_intensity > 135: score += 1.5
    if contrast > 55: score += 1.5
    if edge_strength > 40: score += 1
    if entropy > 6.2: score += 1
    if bright_ratio > 0.18: score += 2  # strong feature

    # MRI indicators (negatively score X-ray)
    if avg_intensity < 120: score -= 1
    if contrast < 48: score -= 1
    if entropy < 5.8: score -= 1

    # -----------------------------
    # Final decision
    # -----------------------------
    if score >= 2:
        return "Chest X-ray"
    else:
        return "Brain MRI"

# -------------------------------
# Predict Disease
# -------------------------------
def predict_disease(image: Image.Image, diagnosis_type: str) -> str:
    img = image.resize((224,224)).convert("RGB")  # match VGG16 input
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if diagnosis_type in ["Brain Tumor Detection", "Brain MRI"]:
        pred = brain_model.predict(img_array)
        return "Tumor Detected 🧠" if pred[0][0] > 0.5 else "No Tumor 👍"

    elif diagnosis_type in ["Chest X-Ray Analysis", "Chest X-ray"]:
        pred = chest_model.predict(img_array)
        return "Pneumonia Detected 😷" if pred[0][0] > 0.5 else "Normal Lungs ✅"
