"""
app.py
Gradio demo: MedQuAD-style medical Q&A with Qwen2.5-1.5B-Instruct + LoRA adapter.
Needs a GPU (4-bit loading), so it runs on Kaggle. Educational use only.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 only; set before torch loads

import gradio as gr
import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prepare_data import SYSTEM_PROMPT  # same prompt as training and evaluation

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "qwen2.5-1.5b-medqa-qlora"  # your HF username is added automatically
MAX_NEW_TOKENS = 250
MAX_QUESTION_CHARS = 500
DEFAULT_REP_PENALTY = 1.15  # demo choice after a 4-question check, not a tuned value

DISCLAIMER = (
    "**Educational use only.** This is a small research demo, not medical advice. "
    "It can be wrong, can invent details or numbers, and sometimes repeats itself. "
    "For personal medical concerns, consult a qualified healthcare professional."
)


def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16
    )
    adapter_repo = f"{HfApi().whoami()['name']}/{ADAPTER_NAME}"
    model = PeftModel.from_pretrained(base, adapter_repo).eval()
    return tok, model


tok, model = load_model()


def answer(question, rep_penalty):
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        return "Please type a question."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            repetition_penalty=float(rep_penalty), pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Your question", lines=3,
                   placeholder="e.g. What are the symptoms of iron deficiency anemia?"),
        gr.Slider(1.0, 1.5, value=DEFAULT_REP_PENALTY, step=0.05,
                  label="Repetition penalty (1.0 = off; higher reduces loops)"),
    ],
    outputs=gr.Textbox(label="Answer", lines=12),
    title="MedQuAD QLoRA Assistant (educational demo)",
    description=DISCLAIMER,
    examples=[
        ["What causes migraines?", DEFAULT_REP_PENALTY],
        ["Is type 2 diabetes hereditary?", DEFAULT_REP_PENALTY],
        ["Do you have information about asthma", DEFAULT_REP_PENALTY],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)