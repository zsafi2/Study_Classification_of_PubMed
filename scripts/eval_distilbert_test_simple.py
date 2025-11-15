import os, csv
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

# 1) Paths
MODEL_DIR = "runs/distilbert_small_local/checkpoint-3760"  # best checkpoint from your log
TEST_CSV  = "data/sample_test_small.csv"
RESULTS   = "results/experiments.csv"

# 2) Load test data
df = pd.read_csv(TEST_CSV).dropna(subset=["title", "label"]).reset_index(drop=True)

# 3) Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to("cpu")
model.eval()

# 4) Run predictions in batches
enc = tokenizer(
    df["title"].tolist(),
    truncation=True,
    max_length=64,
    padding=True,
    return_tensors="pt",
)

preds = []
with torch.no_grad():
    for i in range(0, len(df), 64):
        batch = {k: v[i:i+64] for k, v in enc.items()}
        logits = model(**batch).logits
        preds.append(logits.argmax(dim=1).cpu().numpy())

y_pred = np.concatenate(preds)

# 5) Map predicted ids -> labels using id2label
id2label = model.config.id2label
pred_labels = [id2label[int(i)] for i in y_pred]

# 6) Compute Macro-F1
f1 = f1_score(df["label"], pred_labels, average="macro")
print(f"DistilBERT Test Macro-F1: {f1:.4f}")

# 7) Write / append to results/experiments.csv
os.makedirs("results", exist_ok=True)
header = ["model","train","val","test","metric","score","notes"]
if not os.path.exists(RESULTS):
    with open(RESULTS, "w", newline="") as f:
        csv.writer(f).writerow(header)

with open(RESULTS, "a", newline="") as f:
    csv.writer(f).writerow([
        "distilbert_small",
        "data/sample_train_small.csv",
        "data/sample_val_small.csv",
        TEST_CSV,
        "Test Macro-F1",
        float(f1),
        "best=checkpoint-3760 (local)"
    ])

print("\nSaved to results/experiments.csv")
print(pd.read_csv(RESULTS).to_string(index=False))
