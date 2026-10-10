"""
prepare_data.py
Builds a small, clean, leak-free train/eval split from MedQuAD.
Output: data/train.jsonl, data/eval.jsonl, results/data_stats.json
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "keivalya/MedQuad-MedicalQnADataset"
SEED = 42                  # fixed seed so the split is reproducible
N_EVAL = 200               # held-out examples (never used for training)
N_TRAIN = 2500             # training examples
MIN_ANSWER_CHARS = 50      # drop answers too short to be useful
MAX_ANSWER_CHARS = 1500    # drop very long answers so training fits on a T4
OUT_DIR = "data"
STATS_PATH = "results/data_stats.json"

SYSTEM_PROMPT = (
    "You are a medical information assistant for educational purposes only. "
    "Answer clearly and concisely using general medical knowledge. "
    "You are not a doctor and cannot diagnose or prescribe. "
    "For personal medical concerns, advise the user to consult a qualified "
    "healthcare professional."
)


def clean_text(text):
    """Strip whitespace and collapse repeated spaces/newlines into one space."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_question(text):
    """Same as clean_text, plus fix doubled question marks like '? ?' -> '?'."""
    text = clean_text(text)
    return re.sub(r"\?(\s*\?)+", "?", text)


def normalize(text):
    """Lowercase + clean. Used only to compare texts when looking for duplicates."""
    return clean_text(text).lower()


def load_and_clean():
    """Load the dataset, clean it, drop rows with an empty question or answer."""
    ds = load_dataset(DATASET_NAME, split="train")
    rows_loaded = len(ds)
    rows = []
    for r in ds:
        q = clean_question(r["Question"])
        a = clean_text(r["Answer"])
        if q and a:
            rows.append({"qtype": str(r["qtype"]), "question": q, "answer": a})
    return rows_loaded, rows


def dedupe(rows, field):
    """Keep only the first row for each distinct normalized value of `field`."""
    seen, kept = set(), []
    for r in rows:
        key = normalize(r[field])
        if key not in seen:
            seen.add(key)
            kept.append(r)
    return kept


def length_filter(rows):
    return [r for r in rows if MIN_ANSWER_CHARS <= len(r["answer"]) <= MAX_ANSWER_CHARS]


def split(rows):
    """Shuffle with a fixed seed, then take eval first and train next (no overlap)."""
    need = N_EVAL + N_TRAIN
    if len(rows) < need:
        raise ValueError(f"Only {len(rows)} rows left, but {need} are needed.")
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[N_EVAL:N_EVAL + N_TRAIN], rows[:N_EVAL]  # (train, eval)


def check_no_leakage(train, eval_):
    """Safety check: no question and no answer appears in both sets."""
    tq = {normalize(r["question"]) for r in train}
    ta = {normalize(r["answer"]) for r in train}
    assert not any(normalize(r["question"]) in tq for r in eval_), "Question leak!"
    assert not any(normalize(r["answer"]) in ta for r in eval_), "Answer leak!"


def to_record(r):
    """Store the chat-format messages in the file so train/evaluate/app share one template."""
    return {
        "qtype": r["qtype"],
        "question": r["question"],
        "answer": r["answer"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ],
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_record(r), ensure_ascii=False) + "\n")


def main():
    rows_loaded, rows = load_and_clean()
    n_cleaned = len(rows)

    # Dedupe BEFORE splitting: otherwise a paraphrase of a training question
    # could land in the eval set and make the results look better than they are.
    rows = dedupe(rows, "question")
    n_q = len(rows)
    rows = dedupe(rows, "answer")
    n_a = len(rows)

    rows = length_filter(rows)
    n_len = len(rows)

    train, eval_ = split(rows)
    check_no_leakage(train, eval_)

    Path(OUT_DIR).mkdir(exist_ok=True)
    Path(STATS_PATH).parent.mkdir(exist_ok=True)
    write_jsonl(f"{OUT_DIR}/train.jsonl", train)
    write_jsonl(f"{OUT_DIR}/eval.jsonl", eval_)

    stats = {
        "rows_loaded": rows_loaded,
        "rows_after_cleaning": n_cleaned,
        "rows_after_question_dedupe": n_q,
        "rows_after_answer_dedupe": n_a,
        "rows_after_length_filter": n_len,
        "final_train": len(train),
        "final_eval": len(eval_),
        "constants": {
            "SEED": SEED, "N_EVAL": N_EVAL, "N_TRAIN": N_TRAIN,
            "MIN_ANSWER_CHARS": MIN_ANSWER_CHARS, "MAX_ANSWER_CHARS": MAX_ANSWER_CHARS,
        },
        "qtype_counts_train": dict(Counter(r["qtype"] for r in train)),
        "qtype_counts_eval": dict(Counter(r["qtype"] for r in eval_)),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("loaded:", rows_loaded)
    print("after cleaning:", n_cleaned)
    print("after question dedupe:", n_q)
    print("after answer dedupe:", n_a)
    print("after length filter:", n_len)
    print("train:", len(train), "| eval:", len(eval_))
    print("leakage check passed. Stats saved to", STATS_PATH)


if __name__ == "__main__":
    main()