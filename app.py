import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import sys

# Print system information for debugging
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')}")

# Try to load the model with more detailed error handling
try:
    # First, try the mount path on Streamlit Cloud
    mount_path = '/mount/src/machinelearning_laptoppricepredictor'
    if os.path.exists(f"{mount_path}/pipe.pkl"):
        with open(f"{mount_path}/pipe.pkl", 'rb') as f:
            pipe = pickle.load(f)
        with open(f"{mount_path}/df.pkl", 'rb') as f:
            df = pickle.load(f)
    # If that fails, try the local path
    else:
        with open('pipe.pkl', 'rb') as f:
            pipe = pickle.load(f)
        with open('df.pkl', 'rb') as f:
            df = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Module not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error: {e}")
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