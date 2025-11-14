# src/eval_sklearn_test.py
import argparse, joblib, pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

# usage from a runner script: see run_eval_nb_small.py
def eval_one(model_path: str, test_csv: str):
    pipe = joblib.load(model_path)                # TF–IDF + classifier pipeline
    df = pd.read_csv(test_csv)                    # expects columns: title,label
    pred = pipe.predict(df["title"])
    f1m  = f1_score(df["label"], pred, average="macro")
    acc  = accuracy_score(df["label"], pred)
    print(f"Test Macro-F1: {f1m:.4f} | Accuracy: {acc:.4f}")
    # (Optional) a short per-class report for appendix/debug
    # print(classification_report(df["label"], pred, digits=3))
    return f1m, acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()
    eval_one(args.model, args.test)
