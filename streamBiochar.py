import streamlit as st
import numpy as np
import pickle
import joblib
from  tensorflow.keras.models import load_model

import   streamlit  as st; from PIL import Image; import numpy  as np
import pandas  as pd; import pickle

import os

filename1 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture1.PNG'
filename2 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture2.PNG'

st.title('Predicting Zinc, Cadmium, and Arsenic Levels in European Soils')
with st.container():
    st.image(filename1)
    st.image(filename2)


# # Arrange input boxes into three columns for input features
# col1, col2, col3 = st.columns(3)

# with col1:
#     PHB = st.number_input('Biochar pH', 0.0)
# with col2:
#     Ash = st.number_input('Ash content (%)', 0.0)
# with col3:
#     C_m = st.number_input('Mole contents of C (mmol/g)', 0.0)

# col4, col5, col6 = st.columns(3)
# with col4:
#     H_m = st.number_input('Mole contents of H (mmol/g)', 0.0)
# with col5:
#     N_m = st.number_input('Mole contents of N (mmol/g)', 0.0)
# with col6:
#     O_m = st.number_input('Mole contents of O (mmol/g)', 0.0)

# col7, col8, col9 = st.columns(3)
# with col7:
#     H/C = st.number_input('Atomic ratios (H/C)', 0.0)
# with col8:
#     HO2NC = st.number_input('Atomic ratios (HO2NC)', 0.0)
# with col9:
#     SA = st.number_input('Specific surface area (m2/g)', 0.0)

# col10, col11, col12 = st.columns(3)
# with col10:
#     CEC = st.number_input('Cation exchange capacity of biochar (cmol/kg)', 0.0)
# with col11:
#     AT = st.number_input('Asorption temperature (°C)', 0.0)


# col13, col14, col15 = st.columns(3)
# with col13:
#     PHS = st.number_input('pH of solution)', 0.0)
# with col14:
#     C0 = st.number_input('Initial concentration of heavy metal (mmol/g)', 0.0)
# with col15:
#     HMT = st.number_input('Metal type)', 0.0)


# Arrange input boxes into three columns for input features
col1, col2, col3 = st.columns(3)

with col1:
    PHB = st.number_input('Biochar pH', 0.0)
with col2:
    Ash = st.number_input('Ash content (%)', 0.0)
with col3:
    C_m = st.number_input('Mole contents of C (mmol/g)', 0.0)

col4, col5, col6 = st.columns(3)
with col4:
    H_m = st.number_input('Mole contents of H (mmol/g)', 0.0)
with col5:
    N_m = st.number_input('Mole contents of N (mmol/g)', 0.0)
with col6:
    O_m = st.number_input('Mole contents of O (mmol/g)', 0.0)

col7, col8, col9 = st.columns(3)
with col7:
    H_C = st.number_input('Atomic ratios (H/C)', 0.0)
with col8:
    HO2NC = st.number_input('Atomic ratios (HO2NC)', 0.0)
with col9:
    SA = st.number_input('Specific surface area (m2/g)', 0.0)

col10, col11 = st.columns(2)
with col10:
    CEC = st.number_input('Cation exchange capacity of biochar (cmol/kg)', 0.0)
with col11:
    AT = st.number_input('Adsorption temperature (°C)', 0.0)

col12, col13, col14 = st.columns(3)
with col12:
    PHS = st.number_input('pH of solution', 0.0)
with col13:
    C0 = st.number_input('Initial concentration of heavy metal (mmol/g)', 0.0)

# Dropdown for heavy metal type selection
heavy_metals = ['As3+', 'Cd2+', 'Cu2+', 'Ni2+', 'Pb2+', 'Zn2+']
selected_metal = st.selectbox('Select heavy metal type', heavy_metals)

# Map selected heavy metal to its one-hot encoding
metal_one_hot_map = {
    'As3+': [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    'Cd2+': [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    'Cu2+': [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
    'Ni2+': [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
    'Pb2+': [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
    'Zn2+': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
}
metal_one_hot = metal_one_hot_map[selected_metal]

# Gather all inputs into a list for normalization
input_values = [PHB, Ash, C_m, H_m, N_m, O_m, H_C, HO2NC, SA, CEC, AT, PHS, C0] + metal_one_hot

# Normalize the input values based on min and max values
# Replace with the actual normalization logic based on your dataset
min_values = [6.47, 1.57, 6.3525, 5.2579, 0.0785, -6.7485, 0.0941, -0.6957, 1.15, 0.2123, 20.0, 2.0, 0.0027]
max_values = [12.61, 89.78, 73.4327, 42.7579, 2.3203, 17.0788, 1.076, 0.8893, 485.0, 203.57, 40.0, 10.0, 6.7551]

inputvec = [(2 * (val - min_val) / (max_val - min_val) - 1) if i < len(min_values)
            else val  # Skip normalization for one-hot encoding
            for i, (val, min_val, max_val) in enumerate(zip(input_values, min_values + [0] * 6, max_values + [1] * 6))]
inputvec = np.array(inputvec)



# Check for zeros
zero_count = sum(1 for value in inputvec if value == 0)



# Load models and predict the outputs when the button is pressed
if st.button('Run'):

     ## Validation: If more than 1 inputs are zero, show a warning message
    if zero_count > 1:
        st.error(f"Error: More than one input values are zero. Please provide valid inputs for features.")
    else:

        ## load model
        model2 = joblib.load('model.pkl')

        # YY = stacked_prediction(members, model2, input_values)
        YY = model2.predict(inputvec)


        # Predict removal efficiency
        yhat1 = YY

        # Convert predictions back to the original scale
        RE = (yhat1 + 1) * (100.0 - 0.0) * 0.5 + 0.7  # min=0.7, max=100 for Zinc
        if RE>99: RE=99;
        # Display predictions in a single row using columns
        col19, col20, col21 = st.columns(3)
        # Display predictions
        with col19:
            st.write("removal efficiency (%): ", np.round(abs(RE), decimals=2))


filename7 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture3.PNG'
filename8 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture4.PNG'

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
