import os
os.system("pip install -r requirements.txt")

import streamlit as st
import pickle
import numpy as np

# Load the trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# Use your GitHub-hosted background image
background_image_url = "https://github.com/LavanyaVmk/Laptop-Price-Prediction-ML/blob/main/img1.jpeg?raw=true"

# Apply custom CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: 80%;
        }}
        /* Adjust main container slightly left */
        .block-container {{
            padding-left: 17% !important;
            padding-right: 17% !important;
        }}
        /*  Heading */
        h1 {{
            text-align: left;
            font-size: 14px;
            font-weight: bold;
            background-color: #fffd37 !important;
            width: 48% !important;
            color: #1900ff !important; /* Royal Blue */
        }}
        /* Increase font size for labels */

        /* Change Textbox Label Color to Royal Blue */
           div[data-testid="stSelectbox"] label, 
           div[data-testid="stNumberInput"] label, 
           div[data-testid="stSlider"] label {{
           color: #4169E1 !important;  /* Royal Blue */
           font-weight: bold;
           font-size: 20px;
        }}

        
        /* Adjust Input Field Styles */
        div[data-testid="stSelectbox"], div[data-testid="stNumberInput"], div[data-testid="stSlider"] {{
            width: 45% !important;
            color: black !important;
            font-weight: bold !important;
        }}
        /* Center Predict Button */
        .stButton {{
            display: flex;
            justify-content: center;
        }}
        .stButton > button {{
            background-color: #4169E1 !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 20px;
            font-weight: bold;
            margin-top: 40px !important;
            border: none;
            width: 220px;
            transition: all 0.3s ease-in-out;
        }}
        .stButton > button:hover {{
            background-color: #39FF14 !important;
            color: black !important;
            transform: scale(1.05);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }}
        /* Price Display Box */
        .price-box {{
            background-color: #fffd37;
            padding: 10px;
            border-radius: 8px; 
            text-align: center; 
            font-size: 22px; 
            font-weight: bold;
            width: 50%;
            margin: auto;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Left-aligned title
st.markdown("""<h1>💻 Laptop Price Predictor</h1>""", unsafe_allow_html=True)

# Input Fields
company = st.selectbox('**Brand**', df['Company'].unique())
laptop_type = st.selectbox('**Type**', df['TypeName'].unique())
ram = st.selectbox('**Memory (RAM in GB)**', [2, 4, 6, 8, 12, 16, 24, 32, 64])
touchscreen = st.selectbox('**Touchscreen**', ['No', 'Yes'])
hdd = st.selectbox('**Hard Drive (HDD in GB)**', [0, 128, 256, 512, 1024, 2048])
weight = st.number_input('**Weight (in Kg)**', min_value=0.0, step=0.1)
ips = st.selectbox('**IPS Display**', ['No', 'Yes'])
ssd = st.selectbox('**Solid State Drive (SSD in GB)**', [0, 8, 128, 256, 512, 1024])
screen_size = st.slider('**Screen Size (in inches)**', 10.0, 18.0, 13.0, key='slider_screen_size')
resolution = st.selectbox('**Screen Resolution**', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', 
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('**Processor (CPU)**', df['Cpu brand'].unique())
gpu = st.selectbox('**Graphics Card (GPU)**', df['Gpu brand'].unique())
os = st.selectbox('**Operating System**', df['os'].unique())

# Centered Predict Button
if st.button('Predict Price', key='predict_button'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    price_inr = f"₹{predicted_price:,.2f}"
    
    # Display Price in Styled Box
    st.markdown(f"""<div class='price-box'>🏷️ Estimated Laptop Price: <span style="color: #d80000;">{price_inr}</span></div>""", unsafe_allow_html=True)
