import streamlit as st
import numpy as np

from src.predictions.estimator import train_energy_model, predict_energy
from src.anomaly.detector import flag_anomaly
from src.nlp.complexity_score import get_prompt_complexity
from src.nlp.parser import parse_prompt
from src.optimization.recommender import simplify_prompt
from src.utils.logger import log_usage
from src.gui.layout import input_section, output_section

# ------------------------
# 🚀 Load Models
# ------------------------
model = train_energy_model()

# ------------------------
# 🎛️ GUI & Logic
# ------------------------
st.set_page_config(page_title="Sustainable AI Estimator", layout="centered")
st.title("🔋 Sustainable AI - Energy Estimator & Optimizer")

# Input Widgets
prompt, num_layers, training_hours, flops_per_hour = input_section()

# ------------------------
# 📈 Prediction Pipeline
# ------------------------
if prompt:
    parsed = parse_prompt(prompt)
    complexity = parsed["token_count"]

    # Prediction
    predicted_kwh = predict_energy(model, {
        "layers": num_layers,
        "hours": training_hours,
        "flops": flops_per_hour,
        "complexity": complexity
    })

    # Anomaly Check
    is_anomaly = flag_anomaly([num_layers, training_hours, flops_per_hour, complexity])

    # Optimization
    optimized_prompt = simplify_prompt(prompt)

    # Display Results
    output_section(predicted_kwh, optimized_prompt, is_anomaly)

    # Log for transparency
    log_usage({
        "prompt": prompt,
        "layers": num_layers,
        "hours": training_hours,
        "flops": flops_per_hour,
        "complexity": complexity,
        "predicted_kwh": predicted_kwh,
        "anomaly": is_anomaly
    })