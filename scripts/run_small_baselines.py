# run_small_baselines.py
from src.train_sklearn import train_one

TRAIN = "data/sample_train_small.csv"
VAL   = "data/sample_val_small.csv"

if __name__ == "__main__":
    print("Running TF–IDF baselines on SMALL splits…")
    train_one("logreg",    TRAIN, VAL, "runs/tfidf_logreg_small")
    train_one("linearsvm", TRAIN, VAL, "runs/tfidf_linearsvm_small")
    train_one("nb",        TRAIN, VAL, "runs/tfidf_nb_small")
    print("Done.")
