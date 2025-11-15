import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TRAIN_CSV = "data/sample_train_small.csv"
TEST_CSV  = "data/sample_test_small.csv"
DISTIL_CKPT = "runs/distilbert_small_local/checkpoint-3760"

def get_nb_predictions():
    train = pd.read_csv(TRAIN_CSV).dropna(subset=["title","label"]).reset_index(drop=True)
    test  = pd.read_csv(TEST_CSV).dropna(subset=["title","label"]).reset_index(drop=True)

    X_train = train["title"].astype(str)
    y_train = train["label"].astype(str)
    X_test  = test["title"].astype(str)
    y_test  = test["label"].astype(str)

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=100_000)
    X_train_tf = vec.fit_transform(X_train)
    X_test_tf  = vec.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_tf, y_train)
    y_pred = nb.predict(X_test_tf)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"[NB TF-IDF] Test Macro-F1 (fresh run): {macro_f1:.4f}")

    return test.reset_index(drop=True), y_test.to_numpy(), y_pred, vec, nb

def get_distilbert_predictions(test_df):
    if not os.path.isdir(DISTIL_CKPT):
        raise FileNotFoundError(f"DistilBERT checkpoint not found at {DISTIL_CKPT}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(DISTIL_CKPT)
    model.to("cpu")
    model.eval()

    titles = test_df["title"].astype(str).tolist()
    enc = tokenizer(
        titles,
        truncation=True,
        max_length=64,
        padding=True,
        return_tensors="pt",
    )

    preds = []
    with torch.no_grad():
        for i in range(0, len(titles), 64):
            batch = {k: v[i:i+64] for k, v in enc.items()}
            logits = model(**batch).logits
            preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred_ids = np.concatenate(preds)
    id2label = model.config.id2label
    y_pred_labels = np.array([id2label[int(i)] for i in y_pred_ids])

    macro_f1 = f1_score(test_df["label"], y_pred_labels, average="macro")
    print(f"[DistilBERT] Test Macro-F1 (this script): {macro_f1:.4f}")

    return y_pred_labels

def main():
    # 1) NB predictions
    test_df, y_true, y_nb, vec, nb = get_nb_predictions()

    # 2) DistilBERT predictions
    y_distil = get_distilbert_predictions(test_df)

    # 3) Find interesting cases
    # NB correct, DistilBERT wrong
    nb_correct_distil_wrong = (y_nb == y_true) & (y_distil != y_true)
    # DistilBERT correct, NB wrong
    distil_correct_nb_wrong = (y_distil == y_true) & (y_nb != y_true)

    print("\n=== Examples where NB is correct and DistilBERT is wrong ===\n")
    nb_good = test_df[nb_correct_distil_wrong].copy()
    nb_good["true"] = y_true[nb_correct_distil_wrong]
    nb_good["nb_pred"] = y_nb[nb_correct_distil_wrong]
    nb_good["distil_pred"] = y_distil[nb_correct_distil_wrong]

    for _, row in nb_good.head(5).iterrows():
        print(f"Title       : {row['title']}")
        print(f"True label  : {row['true']}")
        print(f"NB pred     : {row['nb_pred']}")
        print(f"DistilBERT  : {row['distil_pred']}")
        print("-" * 80)

    print("\n=== Examples where DistilBERT is correct and NB is wrong ===\n")
    distil_good = test_df[distil_correct_nb_wrong].copy()
    distil_good["true"] = y_true[distil_correct_nb_wrong]
    distil_good["nb_pred"] = y_nb[distil_correct_nb_wrong]
    distil_good["distil_pred"] = y_distil[distil_correct_nb_wrong]

    for _, row in distil_good.head(5).iterrows():
        print(f"Title       : {row['title']}")
        print(f"True label  : {row['true']}")
        print(f"NB pred     : {row['nb_pred']}")
        print(f"DistilBERT  : {row['distil_pred']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
