
# app.py

import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Insurance Cost Predictor")

st.title("💰 Medical Insurance Cost Prediction")
st.write("Predict insurance cost based on user details")

# Input fields
age = st.slider("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

# Encoding inputs
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

region_dict = {
    "Southwest": 0,
    "Southeast": 1,
    "Northwest": 2,
    "Northeast": 3
}

region = region_dict[region]

# Prediction
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")