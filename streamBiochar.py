import streamlit as st
import numpy as np
import pickle
import joblib
from  tensorflow.keras.models import load_model

import   streamlit  as st; from PIL import Image; import numpy  as np
import pandas  as pd; import pickle

import os

filename1 = 'https://raw.githubusercontent.com/imsb1371/streamBiochar/refs/heads/main/Capture1.PNG'
filename2 = 'https://raw.githubusercontent.com/imsb1371/streamBiochar/refs/heads/main/Capture2.PNG'

st.title('Prediction of Heavy Metal Removal Efficiency Using Biochar')
with st.container():
    st.image(filename1)
    st.image(filename2)



# Arrange input boxes into three columns for input features
col1, col2 = st.columns(2)
with col1:
    ECs = st.number_input('Electrical conductivity (soil, mS·cm-1)', 0.0)
with col2:
    CECs = st.number_input('Cation exchange capacity (Soil, cmol(+)/kg)', 0.0)

col4, col5, col6 = st.columns(3)
with col4:
    Clay = st.number_input('Clay (%)', 0.0)
with col5:
    Silt = st.number_input('Silt (%)', 0.0)
with col6:
    Sand = st.number_input('Sand (%)', 0.0)

col7, col8, col9 = st.columns(3)
with col7:
    OC = st.number_input('Organic carbon (OC, mg·kg-1)', 0.0)
with col8:
    Ptem = st.number_input('Pyrolysis temperature (oC)', 0.0)
with col9:
    CH = st.number_input('Molar ratio of carbon to hydrogen', 0.0)

col10, col11 = st.columns(2)
with col10:
    ONC = st.number_input('Molar ratio of plus of oxygen and nitrogen to carbon ((O + N)/C)', 0.0)

col12, col13 = st.columns(2)
with col12:
    ECb = st.number_input('Electrical conductivity (biochar, mS·cm-1)', 0.0)
with col13:
    CECb = st.number_input('Cation exchange capacity (biochar, cmol(+)/kg)', 0.0)


col15, col16 = st.columns(2)
with col15:
    SSA = st.number_input('Surface area of biochar (m2/g)', 0.0)
with col16:
    THMs = st.number_input('Total Heavy Metals in Soil (mg/kg)', 0.0)

col17, col18 = st.columns(2)
with col17:
    THMb = st.number_input('Total Heavy Metals in Biochar (mg/kg)', 0.0)
with col18:
    AHMs = st.number_input('Available Heavy Metals in Soil (mg/kg))', 0.0)

col19, col20, col21  = st.columns(3)
with col19:
    WHC = st.number_input('Soil water capacity (%)', 0.0)
with col20:
    add = st.number_input('Dosage of biochar (%)', 0.0)
with col21:
    Time = st.number_input('Time (day) ', 0.0)


col22, col23, col24  = st.columns(3)
with col22:
    PHs = st.number_input('pH of soil', 0.0)
with col23:
    PHb = st.number_input('pH of biochar', 0.0)
with col24:
    Ash = st.number_input('Ash content (%)', 0.0)


# Dropdown for heavy metal type selection
col25, col26 = st.columns(2)
with col25:
    heavy_metals = ['As', 'Cd', 'Cu', 'Pb', 'Zn']
    selected_metal = st.selectbox('Select heavy metal type', heavy_metals)

# Map selected heavy metal to its one-hot encoding
metal_one_hot_map = {
    'As': [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    'Cd': [0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
    'Cu': [0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
    'Pb': [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
    'Zn': [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
}
metal_one_hot = metal_one_hot_map[selected_metal]

# Gather all inputs into a list for normalization
input_values = [PHs, ECs, CECs, Clay, Silt, Sand, OC, Ptem, CH, ONC, PHb, ECb, CECb, Ash, SSA, THMs, THMb, AHMs, WHC, add, Time] + metal_one_hot

# Normalize the input values based on min and max values
# Replace with the actual normalization logic based on your dataset

min_values = [2.,0.03,3.96,5.7,3,2.09,680,300,6.74301676,0.005555556,3.25,0.0176,7.42,1.66,0.14,0.03884,0,0,-1,0.5,2]

max_values = [8.64,3.01,24.6,36.96,70,92.3,57600,650,57.3558023,0.923777962,12.2,10.37,96.6,78.7,907.4,32673,1490.6,2600,80,10,150]

numeric_inputs = input_values[:len(min_values)]  # First 21 elements
normalized_inputs = [
    (2 * (val - min_val) / (max_val - min_val) - 1)
    for val, min_val, max_val in zip(numeric_inputs, min_values, max_values)
]

# Combine normalized inputs with one-hot encoding
inputvec = np.array(normalized_inputs + metal_one_hot)

# Check zeros only in the numeric features
zero_count = sum(1 for value in numeric_inputs if value == 0)


if st.button('Run'):
    if zero_count > 1:
        st.error("Error: More than one input values are zero. Please provide valid inputs for features.")
    else:
        try:
            # Load the model
            model2 = joblib.load('Model.pkl')

            # Predict using the model
            inputvec = inputvec.reshape(1, -1)  # Ensure correct shape
            YY = model2.predict(inputvec)

            # Calculate removal efficiency
            RE = (YY + 1) * (100.0 - 0.0) * 0.5 + 0.0
            RE = min(RE, 99)  # Limit RE to 99%

            # Display predictions
            col19, col20, col21 = st.columns(3)
            with col19:
                st.write("Removal efficiency (%): ", np.round(abs(RE), 2))

        except Exception as e:
            st.error(f"Model prediction failed: {e}")



filename7 = 'https://raw.githubusercontent.com/imsb1371/streamBiochar/refs/heads/main/Capture3.PNG'
filename8 = 'https://raw.githubusercontent.com/imsb1371/streamBiochar/refs/heads/main/Capture4.PNG'

col22, col23 = st.columns(2)
with col22:
    with st.container():
        st.markdown("<h5>Developer:</h5>", unsafe_allow_html=True)
        st.image(filename8)

with col23:
    with st.container():
        st.markdown("<h5>Supervisor:</h5>", unsafe_allow_html=True)
        st.image(filename7) 


footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
    This web app was developed in School of Resources and Safety Engineering, Central South University, Changsha 410083, China
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
