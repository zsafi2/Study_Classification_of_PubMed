# src/experiment_nb_coarse.py
#
# Naive Bayes on COARSE labels (broad fields instead of 151 fine labels)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

TRAIN_CSV = "data/sample_train_small.csv"
VAL_CSV   = "data/sample_val_small.csv"
TEST_CSV  = "data/sample_test_small.csv"


def get_field(label: str) -> str:
    """Map full arXiv label to a broad field."""
    if label.startswith("astro-ph"):
        return "astro-ph"
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
    if label.startswith("hep-"):
        return "hep"      # group hep-ex, hep-lat, hep-ph, hep-th
    if label.startswith("nucl-"):
        return "nucl"
    if label.startswith("nlin."):
        return "nlin"
    if label.startswith("gr-qc"):
        return "gr-qc"
    # default = stuff before dot
    return label.split(".")[0]


def load_data():
    train = pd.read_csv(TRAIN_CSV).dropna(subset=["title", "label"])
    val   = pd.read_csv(VAL_CSV).dropna(subset=["title", "label"])
    test  = pd.read_csv(TEST_CSV).dropna(subset=["title", "label"])

    # map fine labels -> coarse fields
    for df in (train, val, test):
        df["coarse_label"] = df["label"].map(get_field)

    return train, val, test


def main():
    print("Loading data...")
    train, val, test = load_data()

    print("Some example coarse labels:")
    print(train[["label", "coarse_label"]].head(10))

    # TF–IDF as before
    print("\nFitting TF–IDF (unigrams + bigrams)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2,
    )
    X_train = tfidf.fit_transform(train["title"])
    X_val   = tfidf.transform(val["title"])
    X_test  = tfidf.transform(test["title"])

    y_train = train["coarse_label"].values
    y_val   = val["coarse_label"].values
    y_test  = test["coarse_label"].values

    # Train NB
    print("\nTraining Multinomial Naive Bayes on coarse labels...")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate on val + test
    print("\n=== Validation performance (coarse labels) ===")
    val_pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, val_pred))
    print(
        classification_report(
            y_val, val_pred, digits=4, zero_division=0
        )
    )

    print("\n=== Test performance (coarse labels) ===")
    test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, test_pred)
    print("Accuracy:", acc)

    report = classification_report(
        y_test,
        test_pred,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    macro_f1 = report["macro avg"]["f1-score"]
    print("Macro-F1:", macro_f1)

    # Show per-coarse-field F1 briefly
    print("\nPer coarse field (test):")
    import pandas as pd
    df_rep = pd.DataFrame(report).T
    print(df_rep.loc[sorted(set(y_test))][["precision","recall","f1-score","support"]])

    # Save to results for later
    df_rep.to_csv("results/nb_coarse_report.csv")
    print("\nSaved full coarse-label report to results/nb_coarse_report.csv")
    print(f"\nSummary -> Test Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
