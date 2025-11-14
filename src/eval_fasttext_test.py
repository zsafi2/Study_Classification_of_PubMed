# src/eval_fasttext_test.py
import argparse, pandas as pd, fasttext

def clean(s):
    # fastText requires one line per example
    return str(s).replace("\n", " ").replace("\r", " ").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test",  required=True)
    args = ap.parse_args()

    m = fasttext.load_model(args.model)
    df = pd.read_csv(args.test).dropna(subset=["title","label"]).reset_index(drop=True)

    titles = [clean(t) for t in df["title"]]
    preds  = [m.predict(t, k=1)[0][0].replace("__label__","") for t in titles]

    acc = (pd.Series(preds).values == df["label"].values).mean()
    print(f"FastText Test Accuracy: {acc:.4f}")

    ex = df.head(5).copy()
    ex["pred"] = preds[:5]
    print("\nExamples:")
    print(ex[["title","label","pred"]].to_string(index=False))

if __name__ == "__main__":
    main()
