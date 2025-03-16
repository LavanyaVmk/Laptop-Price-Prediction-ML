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
st.set_page_config(page_title="Laptop Price Predictor ", layout="wide")

# Use your GitHub-hosted background image
background_image_url = "https://github.com/LavanyaVmk/Laptop-Price-Prediction-ML/blob/main/img1.jpeg?raw=true"

# Apply custom CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: 77%;
        }}
        /* Shift content slightly more to center */
        .block-container {{
            margin-left: 15% !important;
            margin-right: 15% !important;
        }}
        /* Heading in Sunshine Yellow */
        h1 {{
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #fffd37;
        }}
        /* Button Style */
        .stButton {{
            display: flex;
            justify-content: center; /* Slightly more centered */
        }}

        .stButton > button {{
            background-color: black !important;
            color: white !important;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease-in-out;
        }}

        .stButton > button:hover {{
            background-color: #d80000 !important; /* Turns red */
            color: white !important;
            transform: scale(1.1);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }}
        
        /* Input Fields Styling */
        div[data-testid="stSelectbox"], div[data-testid="stNumberInput"] {{
            width: 40% !important;
            margin: auto;
        }}

        /* Adjust slider styling */
        .stSlider {{
            width: 40% !important;
            margin: auto;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Title
st.markdown("""<h1>💻 Laptop Price Predictor </h1>""", unsafe_allow_html=True)

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

# Predict Button (More Centered)
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button('Predict Price', key='predict_button'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    price_inr = f"₹{predicted_price:,.2f}"
    st.markdown(
        f"""
        <div style="background-color: #fffd37; padding: 12px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold;">
            🏷️ Estimated Laptop Price: <span style="color: #d80000;">{price_inr}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)
