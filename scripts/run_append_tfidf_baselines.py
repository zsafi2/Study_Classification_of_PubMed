import os, csv
import pandas as pd

RESULTS = "results/experiments.csv"
os.makedirs("results", exist_ok=True)

# Create file with header if missing
if not os.path.exists(RESULTS):
    with open(RESULTS, "w", newline="") as f:
        csv.writer(f).writerow(
            ["model","train","val","test","metric","score","notes"]
        )

rows = [
    ["tfidf_linearsvm_small",
     "data/sample_train_small.csv",
     "data/sample_val_small.csv",
     "data/sample_test_small.csv",
     "Test Macro-F1",
     0.262259,
     "ngrams=1-2,max_features=100k"],

    ["tfidf_logreg_small",
     "data/sample_train_small.csv",
     "data/sample_val_small.csv",
     "data/sample_test_small.csv",
     "Test Macro-F1",
     0.259532,
     "ngrams=1-2,max_features=100k"],

    ["tfidf_nb_small",
     "data/sample_train_small.csv",
     "data/sample_val_small.csv",
     "data/sample_test_small.csv",
     "Test Macro-F1",
     0.269186,
     "ngrams=1-2,max_features=100k"],
]

with open(RESULTS, "a", newline="") as f:
    csv.writer(f).writerows(rows)

print(pd.read_csv(RESULTS).to_string(index=False))
