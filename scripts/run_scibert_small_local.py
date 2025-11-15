# run_scibert_small_local.py
import os, sys, subprocess

TRAIN = "data/sample_train_small.csv"
VAL   = "data/sample_val_small.csv"
OUT   = "runs/scibert_small_local"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = os.environ.copy()
    # If you don't want GPU/MPS, uncomment:
    # env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable, "src/train_hf.py",
        "--model", "allenai/scibert_scivocab_uncased",
        "--train", TRAIN,
        "--val",   VAL,
        "--out",   OUT,
        "--epochs","2",
        "--patience","1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print("Done. Best checkpoint should be inside:", OUT)
