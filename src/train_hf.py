# src/train_hf.py
import os
import argparse
import numpy as np
import pandas as pd
import torch

from typing import Dict, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate


# ------- Data helpers -------

def load_csv_as_dataset(csv_path: str, label2id: Dict[str, int] | None = None
                        ) -> Tuple[Dataset, Dict[str, int]]:
    """Load CSV with columns [title,label] -> HF Dataset with ['title','labels']."""
    df = pd.read_csv(csv_path).dropna(subset=["title", "label"]).reset_index(drop=True)

    if label2id is None:
        labels = sorted(df["label"].unique().tolist())
        label2id = {lab: i for i, lab in enumerate(labels)}

    df["labels"] = df["label"].map(label2id)
    return Dataset.from_pandas(df[["title", "labels"]], preserve_index=False), label2id


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def tok_fn(batch):
        return tokenizer(batch["title"], truncation=True, max_length=max_len)
    return ds.map(tok_fn, batched=True, remove_columns=["title"])


# ------- Trainer factory -------

def make_trainer(
    model_name: str,
    train_csv: str,
    val_csv: str,
    out_dir: str,
    epochs: int,
    lr: float,
    max_len: int,
    per_device_train_bs: int | None,
    per_device_eval_bs: int | None,
    weight_decay: float,
    max_steps_arg: int,
    patience: int,
    seed: int,
) -> Tuple[Trainer, Dict[int, str]]:
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Datasets
    train_ds, label2id = load_csv_as_dataset(train_csv)
    val_ds, _ = load_csv_as_dataset(val_csv, label2id=label2id)
    id2label = {i: l for l, i in label2id.items()}

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    train_tok = tokenize_dataset(train_ds, tok, max_len)
    val_tok = tokenize_dataset(val_ds, tok, max_len)

    # Metrics
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        f1m = metric_f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
        acc = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
        return {"f1_macro": f1m, "accuracy": acc}

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Batch sizes (auto defaults)
    if per_device_train_bs is None:
        per_device_train_bs = 8 if not torch.cuda.is_available() else 32
    if per_device_eval_bs is None:
        per_device_eval_bs = 16 if not torch.cuda.is_available() else 64

    use_fp16 = torch.cuda.is_available()

    # Some transformers versions require max_steps to be an int (not None).
    # We pass -1 to indicate "use epochs".
    safe_max_steps = max_steps_arg if (max_steps_arg is not None and max_steps_arg > 0) else -1

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        # Note: `evaluation_strategy` is still accepted (deprecated -> 4.46);
        # your install shows a warning but supports it. If you upgrade later,
        # change to `eval_strategy="epoch"`.
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        fp16=use_fp16,
        max_steps=safe_max_steps,
        seed=seed,
        report_to=[],  # no wandb/tensorboard
    )

    callbacks = []
    if patience and patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    return trainer, id2label


# ------- CLI -------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., distilbert-base-uncased or allenai/scibert_scivocab_uncased")
    ap.add_argument("--train", required=True, help="CSV with columns [title,label]")
    ap.add_argument("--val",   required=True, help="CSV with columns [title,label]")
    ap.add_argument("--out",   required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr",     type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=64, help="Title length is short; 64 is enough")
    ap.add_argument("--train_bs", type=int, default=None, help="Auto if None (CPU smaller / GPU larger)")
    ap.add_argument("--eval_bs",  type=int, default=None, help="Auto if None")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=-1, help=">0 caps steps; <=0 uses epochs")
    ap.add_argument("--patience",  type=int, default=2, help="Early stopping patience (epochs); 0 disables")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    trainer, _ = make_trainer(
        model_name=args.model,
        train_csv=args.train,
        val_csv=args.val,
        out_dir=args.out,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
        per_device_train_bs=args.train_bs,
        per_device_eval_bs=args.eval_bs,
        weight_decay=args.weight_decay,
        max_steps_arg=args.max_steps,
        patience=args.patience,
        seed=args.seed,
    )

    trainer.train()
    print("Best checkpoint:", trainer.state.best_model_checkpoint or args.out)


if __name__ == "__main__":
    main()
