
import streamlit as st

def input_section():
    st.subheader("Enter Prompt Configuration")
    prompt = st.text_area("Enter your AI Prompt:")
    num_layers = st.slider("Number of Transformer Layers", 1, 48, 12)
    training_hours = st.slider("Training Time (Hours)", 1, 48, 8)
    flops_per_hour = st.number_input("FLOPs/hour (in GFLOPs)", min_value=1, value=500)
    return prompt, num_layers, training_hours, flops_per_hour

def output_section(predicted_kwh, simplified_prompt, is_anomaly):
    st.markdown(f"### Estimated Energy: {predicted_kwh:.2f} kWh")
    st.markdown("### Suggested Optimized Prompt:")
    st.success(simplified_prompt)
    if is_anomaly:
        st.warning("High energy usage detected — consider using the simplified version.")