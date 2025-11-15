import pandas as pd

df = pd.read_csv("results/experiments.csv")
pivot = df.pivot_table(index="model", columns="metric", values="score", aggfunc="last")

print("Pivot:\n")
print(pivot.to_string())

print("\nMarkdown for your report:\n")
cols = list(pivot.columns)
print("| Model | " + " | ".join(cols) + " |")
print("|---|" + "|".join(["---"]*len(cols)) + "|")
for m, row in pivot.iterrows():
    vals = [f"{row[c]:.4f}" if pd.notna(row[c]) else "" for c in cols]
    print(f"| {m} | " + " | ".join(vals) + " |")
