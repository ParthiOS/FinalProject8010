import streamlit as st
import joblib
import numpy as np
import requests

st.set_page_config(page_title="Sustainable AI Estimator", layout="centered")
st.title("Sustainable AI - Energy Estimator")

# Prompt input
prompt = st.text_area("Prompt (text)", placeholder="e.g., Generate a summary on renewable energy policy...")

# User-configurable parameters
layers = st.slider("Model Layers", 1, 200, 80)
flops = st.number_input("FLOPs per Hour (TFLOPs)", value=250.0)
hours = st.slider("⏱Training Hours", 1, 1000, 100)
region = st.selectbox("Region for Carbon Intensity", ["US", "GB", "DE", "FR", "CA"])

# Dummy carbon intensity
def get_carbon_intensity(region):
    carbon_map = {"US": 400, "GB": 180, "DE": 320, "FR": 90, "CA": 120}
    return carbon_map.get(region, 300)

# Load model
model = joblib.load("co2_predictor.joblib")

if st.button("Estimate Energy Usage"):
    intensity = get_carbon_intensity(region)
    features = np.array([[layers, flops, hours]])
    predicted_co2 = model.predict(features)[0]
    kwh_used = (layers * flops * hours) / 1000
    regional_co2 = kwh_used * intensity / 1000

    st.subheader("Transparency Report")
    st.markdown(f"""
    - Model Layers: {layers}  
    - FLOPs per Hour: {flops} TFLOPs  
    - Training Time: {hours} hours  
    - Estimated Energy: *{kwh_used:.2f} kWh*  
    - Regional Intensity: {intensity} gCO₂/kWh  
    - CO₂ (Predicted by ML Model): *{predicted_co2:.2f} kg*  
    - CO₂ (Region-based Estimate): *{regional_co2:.2f} kg*
    """)