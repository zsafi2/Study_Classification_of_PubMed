import matplotlib.pyplot as plt

models = [
    "TF-IDF + NB",
    "TF-IDF + SVM",
    "TF-IDF + LogReg",
    "fastText",
    "DistilBERT",
    "SciBERT",
]

macro_f1 = [0.2692, 0.2623, 0.2595, 0.0006, 0.1550, 0.2345]

plt.figure(figsize=(6, 3))
plt.bar(models, macro_f1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Macro-F1")
plt.tight_layout()
plt.savefig("results/model_macro_f1.pdf")
plt.close()
