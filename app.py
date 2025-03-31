import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import sys

# Add version check
if sys.version_info >= (3, 12):
    st.warning("Running on Python 3.12+. If you encounter any issues, please contact the administrator.")

# Remove debugging information in production
# st.write(f"Python version: {sys.version}")
# st.write(f"Current working directory: {os.getcwd()}")
# st.write(f"Files in directory: {os.listdir('.')}")

# Simplified model loading
@st.cache_resource
def load_model():
    try:
        with open('pipe.pkl', 'rb') as f:
            pipe = pickle.load(f)
        with open('df.pkl', 'rb') as f:
            df = pickle.load(f)
        return pipe, df
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Python version: " + sys.version)
        return None, None

pipe, df = load_model()

if pipe is None or df is None:
    st.error("Failed to load the model. Please check the model files.")
    st.stop()

st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', 
                         ['1920x1080', '1366x768', '1600x900', '3840x2160', 
                          '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
        
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
        
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    
    # Create a DataFrame with the same feature names as used during training
    query_df = pd.DataFrame([[company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]],
                          columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi',
                                 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    # Predict using the correct feature format
    try:
        predicted_price = pipe.predict(query_df)
        st.title("The predicted price of this configuration is " + str(int(np.exp(predicted_price[0]))))
    except Exception as e:
        st.error(f"Error making prediction: {e}")