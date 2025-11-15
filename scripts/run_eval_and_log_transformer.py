# run_eval_and_log_transformer.py
import os, csv, json, glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

TEST = "data/sample_test_small.csv"
RESULTS = "results/experiments.csv"

def find_best_checkpoint(run_dir: str) -> str | None:
    """Return best checkpoint path if trainer_state.json exists,
    else newest checkpoint-*, else run_dir if it has a model file."""
    if not os.path.isdir(run_dir):
        return None
    # try trainer_state.json
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
    ckpts = [p for p in glob.glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(p)]
    if ckpts:
        ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
        return ckpts[-1]
    # last fallback: if run_dir itself looks like a checkpoint (has config + model)
    if os.path.exists(os.path.join(run_dir, "config.json")):
        return run_dir
    return None

def base_tokenizer_name(ckpt_dir: str, default_name: str) -> str:
    """Read _name_or_path from config.json if available to load the right tokenizer."""
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            name = cfg.get("_name_or_path") or cfg.get("name_or_path")
            if isinstance(name, str) and len(name) > 0:
                return name
        except Exception:
            pass
    return default_name

def eval_macro_f1(model_dir: str, tokenizer_from: str, test_csv: str) -> float:
    df = pd.read_csv(test_csv).dropna(subset=["title", "label"]).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_from)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cpu").eval()

    enc = tok(
        df["title"].tolist(),
        truncation=True,
        max_length=64,
        padding=True,
        return_tensors="pt",
    )

    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            batch = {k: v[i : i + 64] for k, v in enc.items()}
            logits = model(**batch).logits
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    # Map ids -> label strings from the checkpoint config
    id2label = model.config.id2label
    pred_labels = [id2label[int(i)] for i in y_pred]
    return float(f1_score(df["label"], pred_labels, average="macro"))

def append_result(model_name: str, score: float, notes: str):
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w", newline="") as f:
            csv.writer(f).writerow(
                ["model", "train", "val", "test", "metric", "score", "notes"]
            )
    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                model_name,
                "data/sample_train_small.csv",
                "data/sample_val_small.csv",
                "data/sample_test_small.csv",
                "Test Macro-F1",
                score,
                notes,
            ]
        )

def main():
    # DistilBERT
    dist_run = "runs/distilbert_small_local"  # adjust if you named it differently
    dist_ckpt = find_best_checkpoint(dist_run)
    if dist_ckpt is None:
        print(f"[DistilBERT] No checkpoint found under {dist_run}.")
    else:
        tok_name = base_tokenizer_name(dist_ckpt, "distilbert-base-uncased")
        f1 = eval_macro_f1(dist_ckpt, tok_name, TEST)
        print(f"[DistilBERT] {dist_ckpt} | tokenizer={tok_name} | Test Macro-F1: {f1:.4f}")
        # also drop a small file next to the checkpoint for convenience
        with open(os.path.join(dist_ckpt, "TEST_F1.txt"), "w") as f:
            f.write(f"{f1:.6f}\n")
        append_result("distilbert_small", f1, f"best={os.path.basename(dist_ckpt)} (local)")

    # SciBERT (only if you have a local folder for it)
    sci_run = "runs/scibert_small_local"  # change if you trained with another name
    sci_ckpt = find_best_checkpoint(sci_run)
    if sci_ckpt and os.path.isdir(sci_ckpt):
        tok_name = base_tokenizer_name(sci_ckpt, "allenai/scibert_scivocab_uncased")
        f1 = eval_macro_f1(sci_ckpt, tok_name, TEST)
        print(f"[SciBERT] {sci_ckpt} | tokenizer={tok_name} | Test Macro-F1: {f1:.4f}")
        with open(os.path.join(sci_ckpt, "TEST_F1.txt"), "w") as f:
            f.write(f"{f1:.6f}\n")
        append_result("scibert_small", f1, f"best={os.path.basename(sci_ckpt)} (local)")
    else:
        print("[SciBERT] No local checkpoint found; skip.")

    # Show the CSV so you can commit it
    if os.path.exists(RESULTS):
        print("\nCurrent results/experiments.csv:")
        print(pd.read_csv(RESULTS).to_string(index=False))

if __name__ == "__main__":
    main()
