# src/analyze_nb_by_field.py

import os
import pandas as pd

REPORT_CSV = "results/nb_per_class_report.csv"

def get_field(label: str) -> str:
    """
    Map a full arXiv label (like 'cs.AI' or 'astro-ph.CO')
    to a broader field ('cs', 'astro-ph', 'math', etc.).
    """
    # 'astro-ph.CO' -> 'astro-ph'
    if label.startswith("astro-ph"):
        return "astro-ph"
    # 'cond-mat.supr-con' -> 'cond-mat'
    if label.startswith("cond-mat"):
        return "cond-mat"
    if label.startswith("q-bio"):
        return "q-bio"
    if label.startswith("q-fin"):
        return "q-fin"
    if label.startswith("math-ph"):
        return "math-ph"
    if label.startswith("physics"):
        return "physics"
    if label.startswith("cs."):
        return "cs"
    if label.startswith("math."):
        return "math"
    if label.startswith("stat."):
        return "stat"
    if label.startswith("econ."):
        return "econ"
    if label.startswith("eess."):
        return "eess"
    # default: take part before first dot
    return label.split(".")[0]

def main():
    if not os.path.exists(REPORT_CSV):
        print(f"Could not find {REPORT_CSV}")
        return

    df = pd.read_csv(REPORT_CSV, index_col=0)

    # Drop overall rows
    df_classes = df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")

    # Add a 'field' column
    df_classes["field"] = [get_field(lbl) for lbl in df_classes.index]

    # Aggregate by field: average F1 and total support
    by_field = (
        df_classes.groupby("field")
        .agg(
            mean_f1=("f1-score", "mean"),
            total_support=("support", "sum")
        )
        .sort_values("mean_f1", ascending=False)
    )

    print("Naive Bayes performance by broad field:")
    print(by_field.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save to CSV in case you want to use it in the paper later
    out_path = "results/nb_by_field.csv"
    by_field.to_csv(out_path)
    print(f"\nSaved aggregated field stats to {out_path}")

if __name__ == "__main__":
    main()
