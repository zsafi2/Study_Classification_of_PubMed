import os, csv
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import fasttext

MODEL_PATH = "runs/fasttext_small/fasttext_titles.bin"
TEST_CSV   = "data/sample_test_small.csv"
RESULTS    = "results/experiments.csv"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FastText model not found at {MODEL_PATH}")

    # 1) Load test data
    df = pd.read_csv(TEST_CSV).dropna(subset=["title", "label"]).reset_index(drop=True)
    titles = [str(t).replace("\n"," ").replace("\r"," ").strip() for t in df["title"]]
    true_labels = df["label"].tolist()

    # 2) Load FastText model
    model = fasttext.load_model(MODEL_PATH)

    # 3) Get predictions
    preds = []
    for t in titles:
        label = model.predict(t, k=1)[0][0]  # e.g. "__label__cs.AI"
        label = label.replace("__label__", "")
        preds.append(label)

    # 4) Compute metrics
    acc = accuracy_score(true_labels, preds)
    macro_f1 = f1_score(true_labels, preds, average="macro")

    print(f"FastText Test Accuracy: {acc:.4f}")
    print(f"FastText Test Macro-F1: {macro_f1:.4f}")

    # 5) Append Macro-F1 to experiments.csv (we already logged Test Acc before)
    os.makedirs("results", exist_ok=True)
    header = ["model","train","val","test","metric","score","notes"]
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w", newline="") as f:
            csv.writer(f).writerow(header)

    row = [
        "fasttext_small",
        "data/sample_train_small.csv",
        "data/sample_val_small.csv",
        TEST_CSV,
        "Test Macro-F1",
        float(macro_f1),
        "epoch=5,dim=100,ngrams=2 (local)"
    ]

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow(row)

    print("\nUpdated results/experiments.csv:")
    print(pd.read_csv(RESULTS).to_string(index=False))

if __name__ == "__main__":
    main()
