
import gradio as gr
import openai
import time
import os

# Set your OpenAI API key here or as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def estimate_energy(token_count):
    return round(token_count * 0.0003, 5)

def estimate_cost(token_count):
    return round(token_count * 0.00002, 5)

def process_prompt(prompt):
    if not openai.api_key:
        return "⚠️ Please set your OpenAI API key as the environment variable 'OPENAI_API_KEY'.", "", "", "", ""

    try:
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        duration = round(time.time() - start_time, 2)
        reply = response["choices"][0]["message"]["content"]
        total_tokens = response["usage"]["total_tokens"]
        cost = estimate_cost(total_tokens)
        energy = estimate_energy(total_tokens)

        return reply, str(total_tokens), f"{duration} sec", f"${cost}", f"{energy} Wh"

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", ""

interface = gr.Interface(
    fn=process_prompt,
    inputs=gr.Textbox(lines=5, label="Enter Prompt"),
    outputs=[
        gr.Textbox(label="Model Response"),
        gr.Textbox(label="Token Count"),
        gr.Textbox(label="Time Taken"),
        gr.Textbox(label="Estimated Cost"),
        gr.Textbox(label="Estimated Energy Usage")
    ],
    title="Sustainable AI Prompt Analyzer",
    description="Submit a prompt to OpenAI's GPT-3.5. Returns token usage, cost, energy, and processing time."
)

interface.launch()
