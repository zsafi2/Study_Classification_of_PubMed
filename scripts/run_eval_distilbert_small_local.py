# run_eval_distilbert_small_local.py
import subprocess, sys

MODEL_DIR = "runs/distilbert_small_local"
TEST_CSV  = "data/sample_test_small.csv"

if __name__ == "__main__":
    cmd = [sys.executable, "src/eval.py", "--model", MODEL_DIR, "--test", TEST_CSV]
    print("Evaluating:", " ".join(cmd))
    subprocess.run(cmd, check=True)
