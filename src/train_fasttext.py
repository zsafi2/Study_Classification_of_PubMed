# src/train_fasttext.py
import argparse, os, pandas as pd

# Try to import fasttext; suggest a fallback wheel if missing
try:
    import fasttext
except ImportError as e:
    raise SystemExit(
        "fasttext not installed. In VS Code Terminal run:\n"
        "  pip3 install fasttext  ||  pip3 install fasttext-wheel\n"
    )

def to_fasttext_txt(csv_path: str, txt_path: str):
    """Convert CSV (title,label) → fastText supervised format: '__label__<label> <title>'."""
    df = pd.read_csv(csv_path)
    if not {"title","label"}.issubset(df.columns):
        raise ValueError("Expected columns: title,label")
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            title = str(r["title"]).replace("\n", " ").strip()
            label = str(r["label"]).strip()
            f.write(f"__label__{label} {title}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="CSV with columns: title,label")
    ap.add_argument("--val",   required=True, help="CSV with columns: title,label")
    ap.add_argument("--out",   required=True, help="Output dir (will be created)")
    ap.add_argument("--epoch", type=int, default=5, help="Training epochs (default 5 for speed)")
    ap.add_argument("--dim",   type=int, default=100, help="Embedding size (default 100)")
    ap.add_argument("--ngrams",type=int, default=2, help="wordNgrams (1 or 2, default 2)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_txt = os.path.join(args.out, "train.txt")
    val_txt   = os.path.join(args.out, "val.txt")

    print("Preparing fastText files …")
    to_fasttext_txt(args.train, train_txt)
    to_fasttext_txt(args.val,   val_txt)

    print("Training fastText …")
    model = fasttext.train_supervised(
        input=train_txt,
        lr=0.5,
        epoch=args.epoch,
        wordNgrams=args.ngrams,
        minn=2, maxn=5,          # subword info helps with jargon
        dim=args.dim,
        loss="softmax"
    )

    print("Validating …")
    # returns (N, precision@1, recall@1)
    N, p1, r1 = model.test(val_txt)
    print(f"Validation (P@1, R@1, N): ({p1:.4f}, {r1:.4f}, {N})")

    out_path = os.path.join(args.out, "fasttext_titles.bin")
    model.save_model(out_path)
    print("Saved model →", out_path)

if __name__ == "__main__":
    main()
