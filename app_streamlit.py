import os
os.system("pip install scikit-learn")
import sklearn

import streamlit as st
import pickle
import numpy as np
import base64


# Load the trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Laptop Price Predictor - SmartTech Co.", layout="wide")


def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set background
set_background("laptop_background_img.webp")


# App title
st.markdown("<h1>💻 Laptop Price Predictor - SmartTech Co.</h1>", unsafe_allow_html=True)

# Layout - Two columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('**Brand**', df['Company'].unique())
    laptop_type = st.selectbox('**Type**', df['TypeName'].unique())
    ram = st.selectbox('**Memory (RAM in GB)**', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    touchscreen = st.selectbox('**Touchscreen**', ['No', 'Yes'])
    hdd = st.selectbox('**Hard Drive (HDD in GB)**', [0, 128, 256, 512, 1024, 2048])
    weight = st.number_input('**Weight (in Kg)**', min_value=0.0, step=0.1)
    
with col2:
    ips = st.selectbox('**IPS Display**', ['No', 'Yes'])
    ssd = st.selectbox('**Solid State Drive (SSD in GB)**', [0, 8, 128, 256, 512, 1024])
    screen_size = st.slider('**Screen Size (in inches)**', 10.0, 18.0, 13.0)
    resolution = st.selectbox('**Screen Resolution**', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', 
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    cpu = st.selectbox('**Processor (CPU)**', df['Cpu brand'].unique())
    gpu = st.selectbox('**Graphics Card (GPU)**', df['Gpu brand'].unique())
    os = st.selectbox('**Operating System**', df['os'].unique())

# Predict button
if st.button('🔮 Predict Price'):
    # Convert categorical selections
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Compute Pixels Per Inch (PPI)
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create feature array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)

    # Predict price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Convert to INR format
    price_inr = f"₹{predicted_price:,.2f}"

    # Display result with styled box
    st.markdown(
        f"""
        <div style="background-color: #ffcc00; padding: 15px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;">
            🏷️ Estimated Laptop Price: <span style="color: #d80000;">{price_inr}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
