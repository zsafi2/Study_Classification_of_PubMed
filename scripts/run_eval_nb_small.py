# run_eval_nb_small.py (project root)
from src.eval_sklearn_test import eval_one

MODEL = "runs/tfidf_nb_small/model.joblib"
TEST  = "data/sample_test_small.csv"

if __name__ == "__main__":
    print("Evaluating Naive Bayes (TF–IDF) on SMALL test…")
    f1, acc = eval_one(MODEL, TEST)
    # Optional: append to your results log if you created one
    # with open("results/experiments.csv", "a") as f:
    #     f.write(f"tfidf_nb_small,data/sample_train_small.csv,data/sample_val_small.csv,{TEST},Test Macro-F1,{f1:.4f},ngrams=1-2,max_features=100k\n")
