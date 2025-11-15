# run_distilbert_small_local.py
import os, sys, subprocess

TRAIN = "data/sample_train_small.csv"
VAL   = "data/sample_val_small.csv"
OUT   = "runs/distilbert_small_local"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU if no GPU
    cmd = [
        sys.executable, "src/train_hf.py",
        "--model", "distilbert-base-uncased",
        "--train", TRAIN,
        "--val",   VAL,
        "--out",   OUT,
        "--epochs","2",
        "--patience","1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print("Done. Best checkpoint is inside:", OUT)
