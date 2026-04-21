"""
evaluate.py
-----------
Responsible: Riley Bendure

Handles:
  - Running inference on the validation set
  - Computing metrics
  - Saving predictions and error-analysis artifacts
  - Saving subject-wise accuracy and simple figures
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.train import mc_data_collator
from src.utils import ensure_dir

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def get_predictions(
    model: Any,
    dataset: Any,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over the dataset and return predictions and labels."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=mc_data_collator,
        shuffle=False,
    )

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_preds), np.array(all_labels)


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute key classification metrics for a 4-class problem."""
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def save_predictions(
    dataset: Any,
    preds: np.ndarray,
    labels: np.ndarray,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Save per-example predictions and metadata to predictions.csv."""
    ensure_dir(output_dir)

    rows = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        example = dataset.data[i]
        rows.append(
            {
                "idx": i,
                "question": example["question"],
                "option_a": example["opa"],
                "option_b": example["opb"],
                "option_c": example["opc"],
                "option_d": example["opd"],
                "true_label": LABEL_MAP[int(label)],
                "pred_label": LABEL_MAP[int(pred)],
                "correct": int(pred) == int(label),
                "subject": example.get("subject_name", ""),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")
    return df


def error_analysis(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return n correct and n incorrect examples for qualitative analysis."""
    correct_df = df[df["correct"]].head(n)
    incorrect_df = df[~df["correct"]].head(n)
    return correct_df, incorrect_df


def subject_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject accuracy when subject annotations are available."""
    if "subject" not in df.columns or df["subject"].nunique() <= 1:
        print("Subject information not available; skipping subject-wise accuracy.")
        return pd.DataFrame()

    summary = (
        df.groupby("subject")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct_count", "count": "total"})
    )
    summary["accuracy"] = summary["correct_count"] / summary["total"]
    summary = summary.sort_values("accuracy", ascending=False)
    return summary


def plot_subject_accuracy(subject_df: pd.DataFrame, figure_dir: str) -> Optional[str]:
    """Save subject-wise accuracy bar chart and return the file path."""
    if subject_df.empty:
        return None

    ensure_dir(figure_dir)
    top_subjects = subject_df.head(15)

    plt.figure(figsize=(12, 6))
    plt.bar(top_subjects.index.astype(str), top_subjects["accuracy"].values)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("Subject-wise Validation Accuracy (Top 15)")
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.tight_layout()

    plot_path = os.path.join(figure_dir, "subject_accuracy.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Subject-wise accuracy figure saved to: {plot_path}")
    return plot_path


def run_evaluation(
    model: Any,
    val_dataset: Any,
    output_dir: str = "outputs",
    batch_size: int = 16,
    figure_dir: str = "figures",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Run full evaluation pipeline and save all required artifacts."""
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    print("\n=== Running Evaluation ===")
    preds, labels = get_predictions(model, val_dataset, batch_size, device)

    metrics = compute_classification_metrics(preds, labels)
    print(
        "Validation Metrics: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision_macro={metrics['precision_macro']:.4f}, "
        f"recall_macro={metrics['recall_macro']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}"
    )

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    df = save_predictions(val_dataset, preds, labels, output_dir)

    correct_df, incorrect_df = error_analysis(df, n=5)
    correct_df.to_csv(os.path.join(output_dir, "correct_examples.csv"), index=False)
    incorrect_df.to_csv(os.path.join(output_dir, "incorrect_examples.csv"), index=False)

    subj_df = subject_accuracy(df)
    if not subj_df.empty:
        subj_path = os.path.join(output_dir, "subject_accuracy.csv")
        subj_df.to_csv(subj_path)
        print(f"Subject-wise accuracy saved to: {subj_path}")
        plot_subject_accuracy(subj_df, figure_dir)

    return metrics
