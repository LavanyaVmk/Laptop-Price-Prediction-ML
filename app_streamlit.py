import os
os.system("pip install scikit-learn")
import sklearn

import streamlit as st
import numpy as np
import pickle

# Load the trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://raw.githubusercontent.com//Laptop-Price-Prediction-ML/main/laptop_background_img.webp");
        background-size: cover;
        background-position: center;
    }}

    /* Increase font size and bold headings */
    h1, h2, h3, h4, h5, h6 {{
        font-weight: bold;
        font-size: 24px !important;
    }}
    label {{
        font-weight: bold;
        font-size: 18px;
    }}

    /* Center the price prediction box */
    .prediction-box {{
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        background-color: #FFD700;
        border-radius: 10px;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor - SmartTech Co.</h1>", unsafe_allow_html=True)

# User Inputs
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"])
    ram = st.selectbox("Memory (RAM in GB)", [2, 4, 8, 16, 32])
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    hdd = st.selectbox("Hard Drive (HDD in GB)", [0, 500, 1000, 2000])
    weight = st.number_input("Weight (in Kg)", min_value=0.5, max_value=5.0, step=0.1)
    processor = st.selectbox("Processor (CPU)", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"])

with col2:
    laptop_type = st.selectbox("Type", ["Ultrabook", "Gaming", "Notebook", "2 in 1 Convertible", "Netbook"])
    ips_display = st.selectbox("IPS Display", ["No", "Yes"])
    ssd = st.number_input("Solid State Drive (SSD in GB)", min_value=0, max_value=2000, step=128)
    screen_size = st.slider("Screen Size (in inches)", 10.0, 18.0, 13.0)
    resolution = st.selectbox("Screen Resolution", ["1920x1080", "1366x768", "3840x2160"])
    gpu = st.selectbox("Graphics Card (GPU)", ["Intel", "NVIDIA", "AMD"])
    os = st.selectbox("Operating System", ["Windows", "Mac", "Linux"])

# Prediction
if st.button("💰 Predict Price", use_container_width=True):
    # Convert categorical inputs to numerical
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips_display = 1 if ips_display == "Yes" else 0

    # Feature array (Modify according to your model)
    features = np.array([[brand, laptop_type, ram, touchscreen, ips_display, hdd, ssd, screen_size, weight, resolution, processor, gpu, os]])
    
    # Predict price
    predicted_price = model.predict(features)[0]
    
    # Display result
    st.markdown(f"<div class='prediction-box'>💰 Estimated Laptop Price: ₹{int(predicted_price)}</div>", unsafe_allow_html=True)
