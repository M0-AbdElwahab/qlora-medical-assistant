"""
train.py
QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on the MedQuAD train split.
Reads:  data/train.jsonl (made by prepare_data.py)
Writes: logs/train_log.json, logs/train_config.json
Pushes: the LoRA adapter to a PRIVATE Hugging Face model repo.
Never touches data/eval.jsonl (the held-out set).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # use one T4 only; must be set before torch loads

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PATH = "data/train.jsonl"
OUTPUT_DIR = "outputs/adapter"            # local copy of the adapter (git-ignored)
HF_REPO_NAME = "qwen2.5-1.5b-medqa-qlora" # your username is added automatically
LOG_PATH = "logs/train_log.json"
CONFIG_PATH = "logs/train_config.json"

SEED = 42
MAX_LENGTH = 512        # measured: longest training example is 445 tokens
EPOCHS = 1
BATCH_SIZE = 4          # per step, small to fit the T4
GRAD_ACCUM = 4          # 4 x 4 = effective batch of 16
LEARNING_RATE = 2e-4    # typical for LoRA; much higher than full fine-tuning
WARMUP_STEPS = 10
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_train_dataset():
    """Read train.jsonl and split each chat into prompt (system+user) and completion (assistant)."""
    if not Path(TRAIN_PATH).exists():
        raise FileNotFoundError(f"{TRAIN_PATH} not found. Run prepare_data.py first.")
    rows = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            rows.append({"prompt": msgs[:2], "completion": msgs[2:]})
    return Dataset.from_list(rows)


def load_quantized_model():
    """Load the base model in 4-bit (NF4) so it fits in T4 memory."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16 support
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16
    )
    return prepare_model_for_kbit_training(model)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    torch.manual_seed(SEED)
    Path("logs").mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = load_train_dataset()
    print("training examples:", len(dataset))

    model = load_quantized_model()
    lora = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir="outputs/trainer",
        max_length=MAX_LENGTH,
        completion_only_loss=True,      # loss only on the answer, not the prompt
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_strategy="no",             # short run: re-running is simpler than checkpoints
        seed=SEED,
        data_seed=SEED,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tok, peft_config=lora,
    )
    trainable, total = count_params(trainer.model)
    print(f"trainable params: {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    # Save the adapter locally FIRST, so a failed upload never loses the training.
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    config = {
        "base_model": BASE_MODEL, "n_train": len(dataset), "seed": SEED,
        "max_length": MAX_LENGTH, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT, "lora_targets": LORA_TARGETS,
        "trainable_params": trainable, "total_params": total,
        "train_minutes": round(minutes, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("training done in", config["train_minutes"], "min | peak GPU GB:", config["peak_gpu_gb"])

    try:
        username = HfApi().whoami()["name"]
        repo_id = f"{username}/{HF_REPO_NAME}"
        HfApi().create_repo(repo_id, private=True, exist_ok=True)
        trainer.model.push_to_hub(repo_id, private=True)
        print("adapter pushed (private):", repo_id)
    except Exception as e:
        print("PUSH FAILED (adapter is still saved locally in", OUTPUT_DIR, "):", e)


if __name__ == "__main__":
    main()