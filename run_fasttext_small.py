# run_fasttext_small.py
import subprocess, sys, os

TRAIN = "data/sample_train_small.csv"
VAL   = "data/sample_val_small.csv"
OUT   = "runs/fasttext_small"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = [sys.executable, "src/train_fasttext.py",
           "--train", TRAIN, "--val", VAL, "--out", OUT]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done. Model at:", os.path.join(OUT, "fasttext_titles.bin"))
