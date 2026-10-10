"""
evaluate.py
Compares the BASE model with the FINE-TUNED model (base + LoRA adapter)
on the held-out set data/eval.jsonl (never used for training).

Metrics:
  1. Loss / perplexity on the reference answers (teacher-forced, no generation).
  2. ROUGE-L between generated answers and reference answers (greedy decoding).
Writes: results/eval_metrics.json, results/eval_samples.jsonl
Use --limit N for a quick smoke test (files get a "smoke_" prefix).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 only; set before torch loads

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "qwen2.5-1.5b-medqa-qlora"  # your HF username is added automatically
EVAL_PATH = "data/eval.jsonl"
RESULTS_DIR = "results"
MAX_NEW_TOKENS = 300   # generation cap; longer answers are cut off (counted below)
GEN_BATCH = 8          # prompts generated at once


def load_eval(limit):
    with open(EVAL_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def prompt_text(tok, messages):
    """System + user messages rendered in Qwen's chat format, ready for the answer."""
    return tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def answer_loss(model, tok, rows):
    """Average loss per answer token (prompt tokens are masked out) and perplexity."""
    total_nll, total_tokens = 0.0, 0
    for r in rows:
        p_ids = tok(prompt_text(tok, r["messages"]), add_special_tokens=False)["input_ids"]
        a_ids = tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + a_ids], device=model.device)
        labels = ids.clone()
        labels[0, :len(p_ids)] = -100  # ignore the prompt; score only the answer
        loss = model(input_ids=ids, labels=labels).loss
        total_nll += loss.item() * len(a_ids)
        total_tokens += len(a_ids)
    avg = total_nll / total_tokens
    return avg, math.exp(avg)


@torch.no_grad()
def generate_answers(model, tok, rows):
    """Greedy decoding, so the same model always gives the same answer."""
    outs = []
    for i in range(0, len(rows), GEN_BATCH):
        texts = [prompt_text(tok, r["messages"]) for r in rows[i:i + GEN_BATCH]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  generated {min(i + GEN_BATCH, len(rows))}/{len(rows)}", flush=True)
    return outs


def rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds, refs)]


def mean(xs):
    return sum(xs) / len(xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N examples")
    args = parser.parse_args()

    rows = load_eval(args.limit)
    refs = [r["answer"] for r in rows]
    prefix = "smoke_" if args.limit else ""
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    print("held-out examples:", len(rows))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"  # decoder-only models must be padded on the left for generation

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16
    )
    adapter_repo = f"{HfApi().whoami()['name']}/{ADAPTER_NAME}"
    model = PeftModel.from_pretrained(base, adapter_repo)
    model.eval()
    print("adapter loaded from:", adapter_repo)

    start = time.time()
    metrics, preds_by, scores_by = {}, {}, {}
    for name in ("base", "finetuned"):
        print(f"\n=== evaluating: {name} ===")
        # "base" = the same model with the adapter switched off
        ctx = model.disable_adapter() if name == "base" else nullcontext()
        with ctx:
            loss, ppl = answer_loss(model, tok, rows)
            preds = generate_answers(model, tok, rows)
        scores = rouge_l(preds, refs)
        preds_by[name], scores_by[name] = preds, scores
        metrics[name] = {
            "loss": round(loss, 4),
            "perplexity": round(ppl, 3),
            "rougeL_mean": round(mean(scores), 4),
            "mean_generated_words": round(mean([len(p.split()) for p in preds]), 1),
            "answers_hitting_token_cap_approx": sum(
                len(tok(p, add_special_tokens=False)["input_ids"]) >= MAX_NEW_TOKENS - 1 for p in preds
            ),
        }
        print(metrics[name])

    metrics["reference_mean_words"] = round(mean([len(r.split()) for r in refs]), 1)
    metrics["finetuned_beats_base_rougeL_count"] = sum(
        f > b for f, b in zip(scores_by["finetuned"], scores_by["base"])
    )
    metrics["n_eval"] = len(rows)
    metrics["max_new_tokens"] = MAX_NEW_TOKENS
    metrics["adapter_repo"] = adapter_repo
    metrics["eval_minutes"] = round((time.time() - start) / 60, 2)

    with open(f"{RESULTS_DIR}/{prefix}eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{RESULTS_DIR}/{prefix}eval_samples.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "question": r["question"], "reference": r["answer"],
                "base": preds_by["base"][i], "finetuned": preds_by["finetuned"][i],
                "rougeL_base": round(scores_by["base"][i], 4),
                "rougeL_finetuned": round(scores_by["finetuned"][i], 4),
            }, ensure_ascii=False) + "\n")
    print("\nsaved:", f"{RESULTS_DIR}/{prefix}eval_metrics.json", "and", f"{RESULTS_DIR}/{prefix}eval_samples.jsonl")


if __name__ == "__main__":
    main()