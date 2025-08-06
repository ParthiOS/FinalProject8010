import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_processing import load_energy_data, preprocess_data
from src.predict import predict_energy

st.set_page_config(page_title="Prompt Optimizer & Energy Estimator", layout="wide")
st.title("Sustainable AI - Prompt Optimizer & Energy Estimator")

# Sidebar Inputs
st.sidebar.header("Model Parameters")
prompt = st.sidebar.text_area("Enter AI Prompt", help="Paste your AI prompt here.")
layers = st.sidebar.slider("Number of Layers", 1, 48, 24)
hours = st.sidebar.slider("Training Time (hrs)", 1, 48, 12)
flops = st.sidebar.number_input("FLOPs per hour (GFLOPs)", min_value=10, value=750)

# Simulated Prompt Optimizer (Dummy Rewrites)
def generate_dummy_optimized_prompts(original_prompt):
    words = original_prompt.split()
    variants = [
        ' '.join(words[:int(len(words)*0.8)]),
        ' '.join(words[:int(len(words)*0.6)]),
        ' '.join(words[:int(len(words)*0.7)]),
        ' '.join(words[:int(len(words)*0.5)]),
        ' '.join(words[:int(len(words)*0.9)]),
    ]
    return [v if v else original_prompt for v in variants]

# Exploratory Data Analysis (EDA) for a Prompt
def eda_for_prompt(prompt_text):
    word_count = len(prompt_text.split())
    char_count = len(prompt_text)
    avg_word_length = np.mean([len(word) for word in prompt_text.split()]) if word_count > 0 else 0
    return word_count, char_count, avg_word_length

# Main Action
if st.button("Optimize & Estimate Energy"):
    if not prompt.strip():
        st.error("Please enter a prompt first.")
    else:
        # Original Prompt EDA
        orig_word_count, orig_char_count, orig_avg_word_len = eda_for_prompt(prompt)

        # Original Energy Estimation
        complexity = orig_word_count
        original_kwh = predict_energy("co2_predictor.joblib", layers, hours, flops, complexity)

        # Generate Optimized Prompts (Simulated)
        optimized_prompts = generate_dummy_optimized_prompts(prompt)

        st.subheader("Original Prompt EDA")
        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", orig_word_count)
        col2.metric("Character Count", orig_char_count)
        col3.metric("Avg Word Length", f"{orig_avg_word_len:.2f}")

        # Pie Chart for Character Distribution
        fig_pie = go.Figure(data=[go.Pie(labels=['Spaces', 'Characters'],
                                         values=[prompt.count(' '), orig_char_count - prompt.count(' ')],
                                         hole=.3)])
        fig_pie.update_layout(title="Character Composition")
        st.plotly_chart(fig_pie)

        st.subheader("Optimized Prompt Suggestions and Comparison")

        results = []
        best_efficiency = 0
        best_prompt = ""

        for idx, opt_prompt in enumerate(optimized_prompts):
            opt_word_count, opt_char_count, opt_avg_word_len = eda_for_prompt(opt_prompt)
            opt_kwh = predict_energy("co2_predictor.joblib", layers, hours, flops, opt_word_count)
            energy_saved = original_kwh - opt_kwh
            efficiency_score = (energy_saved / original_kwh) * 100 if original_kwh != 0 else 0

            results.append({
                'Suggestion': f"Suggestion {idx+1}",
                'Prompt': opt_prompt,
                'Word Count': opt_word_count,
                'Character Count': opt_char_count,
                'Avg Word Length': opt_avg_word_len,
                'Energy_kWh': opt_kwh,
                'Efficiency': efficiency_score
            })

            if efficiency_score > best_efficiency:
                best_efficiency = efficiency_score
                best_prompt = opt_prompt

        df_results = pd.DataFrame(results)

        # KPI Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Energy (kWh)", f"{original_kwh:.2f}")
        col2.metric("Best Efficiency Gain (%)", f"{best_efficiency:.2f}%")
        col3.metric("Layers", layers)

        # Results Table
        st.dataframe(df_results)

        # Bar Chart Comparison (All Variants)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_results['Suggestion'],
            y=df_results['Energy_kWh'],
            name='Optimized Prompts',
            marker_color='lightgreen'
        ))
        fig_bar.add_trace(go.Bar(
            x=['Original'],
            y=[original_kwh],
            name='Original Prompt',
            marker_color='indianred'
        ))
        fig_bar.update_layout(title="Energy Consumption Comparison", barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Energy Efficiency Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best_efficiency,
            title={'text': "Efficiency Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge)

        # Best Prompt Highlight
        st.success(f"Best Optimized Prompt (Efficiency +{best_efficiency:.2f}%):")
        st.info(best_prompt)

# Data Preview (Optional)
st.subheader("Sample Data Preview")
df = load_energy_data("data/model_energy_data.csv")
df = preprocess_data(df)
st.dataframe(df.head())
