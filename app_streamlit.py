import os
os.system("pip install scikit-learn")
import sklearn

import streamlit as st
import pickle
import numpy as np

# Load the trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Apply custom styles
st.markdown("""
    <style>
    body {
        background-image: url("https://source.unsplash.com/1600x900/?technology,laptop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Arial', sans-serif;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }

    h1 {
        font-size: 34px !important;
        color: #003366;
        text-align: center;
        font-weight: bold;
        animation: fadeIn 2s ease-in-out;
    }

    h2, h3 {
        font-size: 24px !important;
        color: #003366;
        font-weight: bold;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 12px;
        border-radius: 8px;
        width: 100%;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #f0f0f0;
        border-radius: 5px;
        padding: 5px;
    }

    .prediction-box {
        background-color: #F4D03F;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# App title with company branding
st.title("💻 Laptop Price Predictor - SmartTech Co.")

# Arrange input fields in two columns for better layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    ram = st.selectbox('Memory (RAM in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    hdd = st.selectbox('Hard Drive (HDD in GB)', [0, 128, 256, 512, 1024, 2048])
    weight = st.number_input('Weight (in Kg)')

with col2:
    laptop_type = st.selectbox('Type', df['TypeName'].unique())
    ips = st.selectbox('IPS Display', ['No', 'Yes'])
    ssd = st.selectbox('Solid State Drive (SSD in GB)', [0, 8, 128, 256, 512, 1024])
    screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', 
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])

cpu = st.selectbox('Processor (CPU)', df['Cpu brand'].unique())
gpu = st.selectbox('Graphics Card (GPU)', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# Predict price
if st.button('💰 Predict Price'):
    # Convert categorical selections
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Compute Pixels Per Inch (PPI)
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create feature array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)

    # Predict price and format output
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Display result in a better format
    st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color:#2E4053;">💰 Estimated Laptop Price: <span style="color:#E74C3C;">₹{predicted_price}</span></h2>
        </div>
    """, unsafe_allow_html=True)



