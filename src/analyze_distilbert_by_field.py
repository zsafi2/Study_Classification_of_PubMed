# src/analyze_distilbert_by_field.py

import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

TEST_CSV = "data/sample_test_small.csv"
DISTIL_RUN = "runs/distilbert_small_local"  # same as before
OUT_PER_CLASS = "results/distilbert_per_class_report.csv"
OUT_BY_FIELD = "results/distilbert_by_field.csv"

def find_best_checkpoint(run_dir: str) -> str | None:
    """Return best checkpoint path if trainer_state.json exists,
    else newest checkpoint-*, else run_dir if it has a model file."""
    if not os.path.isdir(run_dir):
        return None

    st_path = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(st_path):
        try:
            with open(st_path) as f:
                st = json.load(f)
            cp = st.get("best_model_checkpoint")
            if cp and os.path.isdir(cp):
                return cp
        except Exception:
            pass

    # fallback: latest checkpoint-*
    ckpts = [p for p in os.listdir(run_dir) if p.startswith("checkpoint-")]
    if ckpts:
        ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
        return os.path.join(run_dir, ckpts_sorted[-1])

    # last fallback: if run_dir itself has config+model
    if os.path.exists(os.path.join(run_dir, "config.json")):
        return run_dir

    return None

def get_field(label: str) -> str:
    """Same mapping as for Naive Bayes per-field analysis."""
    if label.startswith("astro-ph"):
        return "astro-ph"
    if label.startswith("cond-mat"):
        return "cond-mat"
    if label.startswith("q-bio"):
        return "q-bio"
    if label.startswith("q-fin"):
        return "q-fin"
    if label.startswith("math-ph"):
        return "math-ph"
    if label.startswith("physics"):
        return "physics"
    if label.startswith("cs."):
        return "cs"
    if label.startswith("math."):
        return "math"
    if label.startswith("stat."):
        return "stat"
    if label.startswith("econ."):
        return "econ"
    if label.startswith("eess."):
        return "eess"
    if label.startswith("hep-"):
        # group hep-ex, hep-lat, hep-ph, hep-th
        return "hep"
    if label.startswith("nucl-"):
        return "nucl"
    if label.startswith("nlin."):
        return "nlin"
    if label.startswith("gr-qc"):
        return "gr-qc"
    # default: prefix before dot
    return label.split(".")[0]

def main():
    os.makedirs("results", exist_ok=True)

    # 1) Find best checkpoint
    ckpt = find_best_checkpoint(DISTIL_RUN)
    if ckpt is None:
        print(f"Could not find a checkpoint under {DISTIL_RUN}")
        return
    print(f"Using checkpoint: {ckpt}")

    # 2) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # 3) Load test data
    df = pd.read_csv(TEST_CSV).dropna(subset=["title", "label"]).reset_index(drop=True)
    texts = df["title"].tolist()
    true_labels = df["label"].tolist()

    # 4) Predict in batches
    preds_ids = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=64,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_ids.extend(batch_preds)

    preds_ids = np.array(preds_ids)

    # map ids -> label strings from model config
    id2label = model.config.id2label
    pred_labels = [id2label[int(i)] for i in preds_ids]

    # 5) Per-class classification report
    report = classification_report(
        true_labels,
        pred_labels,
        labels=sorted(set(true_labels)),  # ensure all test labels included
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report).T
    df_report.to_csv(OUT_PER_CLASS)
    print(f"Saved per-class report to {OUT_PER_CLASS}")

    # 6) Aggregate by field
    # Filter out overall rows if they appear (accuracy, macro avg, weighted avg)
    df_classes = df_report.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore").copy()
    df_classes["field"] = [get_field(lbl) for lbl in df_classes.index]

    by_field = (
        df_classes.groupby("field")
        .agg(
            mean_f1=("f1-score", "mean"),
            total_support=("support", "sum"),
        )
        .sort_values("mean_f1", ascending=False)
    )

    by_field.to_csv(OUT_BY_FIELD)
    print("\nDistilBERT performance by broad field:")
    print(by_field.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved aggregated field stats to {OUT_BY_FIELD}")

if __name__ == "__main__":
    main()
