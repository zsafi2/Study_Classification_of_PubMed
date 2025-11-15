import os, csv
import pandas as pd

RESULTS = "results/experiments.csv"
os.makedirs("results", exist_ok=True)

# Make sure file exists with header
if not os.path.exists(RESULTS):
    with open(RESULTS, "w", newline="") as f:
        csv.writer(f).writerow(
            ["model","train","val","test","metric","score","notes"]
        )

rows = [
    # SciBERT from Colab
    ["scibert_small",
     "data/sample_train_small.csv",
     "data/sample_val_small.csv",
     "data/sample_test_small.csv",
     "Test Macro-F1",
     0.2345,            # from Colab
     "epochs=2 (Colab)"],

    # FastText from Colab
    ["fasttext_small",
     "data/sample_train_small.csv",
     "data/sample_val_small.csv",
     "data/sample_test_small.csv",
     "Test Acc",
     0.0135,            # from Colab
     "epoch=5,dim=100,ngrams=2 (Colab)"],
]

with open(RESULTS, "a", newline="") as f:
    csv.writer(f).writerows(rows)

print(pd.read_csv(RESULTS).to_string(index=False))
