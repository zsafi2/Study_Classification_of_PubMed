# src/analyze_nb_per_class.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

TRAIN_CSV = "data/sample_train_small.csv"
VAL_CSV   = "data/sample_val_small.csv"   # we won't use val here, but kept for reference
TEST_CSV  = "data/sample_test_small.csv"

def load_data():
    train = pd.read_csv(TRAIN_CSV).dropna(subset=["title", "label"])
    test  = pd.read_csv(TEST_CSV).dropna(subset=["title", "label"])
    return train, test

def main():
    print("Loading data...")
    train, test = load_data()

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    y_test  = le.transform(test["label"])

    # TF–IDF: unigrams + bigrams, like before
    print("Fitting TF–IDF vectorizer...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2
    )
    X_train = tfidf.fit_transform(train["title"])
    X_test  = tfidf.transform(test["title"])

    # Train Naive Bayes
    print("Training Multinomial Naive Bayes...")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict on test
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    # Overall metrics + per-class metrics
    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    # Convert to DataFrame for easy viewing
    df_report = pd.DataFrame(report).T

    # Save full report
    out_path = "results/nb_per_class_report.csv"
    os.makedirs("results", exist_ok=True)
    df_report.to_csv(out_path)
    print(f"Saved full per-class report to {out_path}")

    # Show a summary: top 5 best and worst labels by F1
    per_class = df_report.iloc[:-3]  # drop 'accuracy', 'macro avg', 'weighted avg'
    per_class_sorted = per_class.sort_values("f1-score", ascending=False)

    print("\nTop 5 best labels by F1:")
    print(per_class_sorted.head(5)[["precision", "recall", "f1-score", "support"]])

    print("\nTop 5 worst labels by F1:")
    print(per_class_sorted.tail(5)[["precision", "recall", "f1-score", "support"]])

if __name__ == "__main__":
    import os
    main()
