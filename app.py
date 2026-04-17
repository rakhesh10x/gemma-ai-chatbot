import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_IDS = {
    "Base Model (gemma-2b)":           "google/gemma-2b",
    "Instruction Model (gemma-2b-it)": "google/gemma-2b-it",
}

HF_TOKEN = os.environ.get("HF_TOKEN", None)

if torch.cuda.is_available():
    DTYPE = torch.float16
    DEVICE = "cuda"
    print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DTYPE = torch.float32
    DEVICE = "cpu"
    print("[INFO] No GPU — using CPU with float32")

NUM_THREADS = os.cpu_count() or 4
torch.set_num_threads(NUM_THREADS)

_model_cache: dict = {}

def load_model(model_key: str):
    if model_key in _model_cache:
        return _model_cache[model_key]["tokenizer"], _model_cache[model_key]["model"]

    model_id = MODEL_IDS[model_key]
    print(f"[INFO] Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
    )

    model.eval()
    _model_cache[model_key] = {"tokenizer": tokenizer, "model": model}
    print(f"[INFO] Model loaded successfully ✅")
    return tokenizer, model

def format_prompt_it(user_prompt: str) -> str:
    return (
        f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def generate_response(model_choice, user_prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    if not user_prompt or not user_prompt.strip():
        return "⚠️ Please enter a prompt before clicking Generate."

    try:
        tokenizer, model = load_model(model_choice)
    except Exception as e:
        return f"❌ Model loading error:\n\n{e}"

    is_instruction = "Instruction" in model_choice
    prompt = format_prompt_it(user_prompt) if is_instruction else user_prompt

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
    except Exception as e:
        return f"❌ Generation error:\n\n{e}"

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip() or "(No output — try a different prompt.)"

CSS = """
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #666; margin-bottom: 16px; }
"""

with gr.Blocks(css=CSS, title="Gemma 2B Chatbot") as demo:

    gr.Markdown("# 🤖 Gemma 2B Chatbot", elem_id="title")
    gr.Markdown("Compare **Base** vs **Instruction-tuned** Gemma 2B models in one place.", elem_id="subtitle")

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_IDS.keys()),
                value=list(MODEL_IDS.keys())[0],
                label="🧠 Select Model",
                info="Base model continues text; Instruction model follows instructions.",
            )
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Generation Settings", open=False):
                max_tokens_slider = gr.Slider(minimum=50, maximum=512, value=100, step=10, label="Max New Tokens")
                temperature_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(lines=5, placeholder="Enter your prompt here…", label="📝 Your Prompt")
            with gr.Row():
                submit_btn = gr.Button("🚀 Generate", variant="primary")
                clear_btn  = gr.Button("🗑️ Clear")
        with gr.Column():
            output_box = gr.Textbox(lines=10, label="Response", interactive=False)

    gr.Examples(
        examples=[
            ["Base Model (gemma-2b)",           "Once upon a time in a distant galaxy,"],
            ["Instruction Model (gemma-2b-it)", "Explain quantum computing in simple terms."],
            ["Instruction Model (gemma-2b-it)", "Write a short Python function to reverse a string."],
        ],
        inputs=[model_dropdown, prompt_input],
        label="📚 Try These Examples",
    )

    submit_btn.click(fn=generate_response, inputs=[model_dropdown, prompt_input, max_tokens_slider, temperature_slider, top_p_slider], outputs=output_box)
    prompt_input.submit(fn=generate_response, inputs=[model_dropdown, prompt_input, max_tokens_slider, temperature_slider, top_p_slider], outputs=output_box)
    clear_btn.click(fn=lambda: ("", ""), outputs=[prompt_input, output_box])

demo.launch(share=True)
