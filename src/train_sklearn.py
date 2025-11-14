# src/train_sklearn.py
import os, joblib, argparse, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

def get_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, n_jobs=None)
    if name == "linearsvm":
        return LinearSVC()
    if name == "nb":
        return MultinomialNB()
    raise ValueError(f"Unknown model: {name}")

def train_one(model_name: str, train_csv: str, val_csv: str, out_dir: str) -> float:
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_csv(train_csv)  # expects columns: title,label
    va = pd.read_csv(val_csv)

    vec = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),     # uni + bi-grams (good for titles)
        min_df=2,
        max_features=100_000    # cap for speed
    )
    clf = get_model(model_name)
    pipe = make_pipeline(vec, clf)

    pipe.fit(tr["title"], tr["label"])
    pred = pipe.predict(va["title"])
    f1m = f1_score(va["label"], pred, average="macro")

    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "VAL_F1.txt"), "w") as f:
        f.write(f"{f1m:.6f}")

    print(f"{model_name} | Val Macro-F1: {f1m:.4f} | saved → {out_dir}")
    return f1m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["logreg","linearsvm","nb"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train_one(args.model, args.train, args.val, args.out)
