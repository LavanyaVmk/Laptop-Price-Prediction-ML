import os
os.system("pip install scikit-learn")
import sklearn

import streamlit as st
import pickle
import numpy as np

# Load the trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Laptop Price Predictor - SmartTech Co.", layout="wide")

# Background image URL from GitHub
background_image_url = "https://github.com/LavanyaVmk/Laptop-Price-Prediction-ML/blob/main/img2.jpg?raw=true"

# Apply CSS for custom styling
st.markdown(
    f"""
    <style>
        /* Background styling */
        .stApp {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: contain;
        }}

        /* Remove grid layout */
        .block-container {{
            padding: 2rem;
            max-width: 800px;
            margin: auto;
        }}

        /* Input fields - Gray background */
        .stTextInput, .stSelectbox, .stNumberInput, .stSlider {{
            background-color: #f0f0f0 !important;
            border-radius: 8px;
        }}

        /* Center button */
        .button-container {{
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }}

        /* Button styling - Light aquatic */
        .stButton > button {{
            background-color: #80d4ff !important;  /* Light aquatic */
            color: black !important;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 20px;
        }}

        /* Estimated Price Section */
        .price-container {{
            background-color: #ffcc00;
            color: black;
            font-size: 20px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin-top: 20px;
        }}

        /* Rupee symbol color */
        .rupee {{
            color: #2ecc71;  /* Light green */
            font-weight: bold;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Title
st.markdown("<h1 style='text-align:center;'>💻 Laptop Price Predictor - SmartTech Co.</h1>", unsafe_allow_html=True)




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
