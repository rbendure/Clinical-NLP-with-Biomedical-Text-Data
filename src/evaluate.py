"""
evaluate.py
-----------
Responsible: Riley Bendure

Handles:
  - Running inference on the validation set
  - Computing accuracy
  - Saving predictions to CSV
  - Printing and saving error analysis (5 correct + 5 incorrect examples)
  - Optional: subject-wise accuracy breakdown
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.train import mc_data_collator
from src.utils import ensure_dir

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_predictions(model, dataset, batch_size: int = 16, device=None):
    """
    Run inference over the entire dataset and collect predictions.

    Parameters
    ----------
    model      : Trained multiple-choice model
    dataset    : MedMCQADataset instance
    batch_size : Inference batch size
    device     : torch.device (auto-detected if None)

    Returns
    -------
    all_preds  : np.ndarray of predicted class indices (0-3)
    all_labels : np.ndarray of true class indices (0-3)
    """
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

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, 4)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Return overall accuracy as a float between 0 and 1."""
    return float(np.mean(preds == labels))


# ---------------------------------------------------------------------------
# Save predictions CSV
# ---------------------------------------------------------------------------

def save_predictions(
    dataset,
    preds: np.ndarray,
    labels: np.ndarray,
    output_dir: str = "outputs",
):
    """
    Save predictions, true labels, and question text to a CSV file.

    Parameters
    ----------
    dataset    : MedMCQADataset (raw HF dataset accessible via .data)
    preds      : Predicted class indices
    labels     : True class indices
    output_dir : Directory to write predictions.csv
    """
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


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(df: pd.DataFrame, n: int = 5):
    """
    Print and return n correct + n incorrect examples for qualitative review.

    Parameters
    ----------
    df : DataFrame produced by save_predictions()
    n  : Number of examples per category (default 5)

    Returns
    -------
    correct_df, incorrect_df : DataFrames with the sampled examples
    """
    correct_df = df[df["correct"] == True].head(n)
    incorrect_df = df[df["correct"] == False].head(n)

    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS  –  {n} Correct  |  {n} Incorrect examples")
    print("=" * 60)

    def _print_examples(subset, tag):
        print(f"\n--- {tag} ---")
        for _, row in subset.iterrows():
            print(f"  Q : {row['question'][:100]}...")
            print(f"  A : {row['option_a']}  B : {row['option_b']}")
            print(f"  C : {row['option_c']}  D : {row['option_d']}")
            print(f"  True: {row['true_label']}  |  Pred: {row['pred_label']}")
            print()

    _print_examples(correct_df, "CORRECT")
    _print_examples(incorrect_df, "INCORRECT")

    return correct_df, incorrect_df


# ---------------------------------------------------------------------------
# Subject-wise accuracy (bonus)
# ---------------------------------------------------------------------------

def subject_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subject accuracy from the predictions DataFrame.

    Returns a DataFrame sorted by accuracy (descending).
    """
    if df["subject"].nunique() <= 1:
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


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    model,
    val_dataset,
    output_dir: str = "outputs",
    batch_size: int = 16,
    device=None,
):
    """
    End-to-end evaluation: inference → metrics → save → error analysis.

    Parameters
    ----------
    model       : Trained multiple-choice model
    val_dataset : MedMCQADataset (validation split)
    output_dir  : Where to save results
    batch_size  : Inference batch size
    device      : torch.device

    Returns
    -------
    metrics : dict with "accuracy" key
    """
    ensure_dir(output_dir)

    print("\n=== Running Evaluation ===")
    preds, labels = get_predictions(model, val_dataset, batch_size, device)

    accuracy = compute_accuracy(preds, labels)
    metrics = {"accuracy": accuracy}

    print(f"\nValidation Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save predictions CSV
    df = save_predictions(val_dataset, preds, labels, output_dir)

    # Error analysis (print + capture results for saving)
    correct_df, incorrect_df = error_analysis(df, n=5)
    correct_df.to_csv(os.path.join(output_dir, "correct_examples.csv"), index=False)
    incorrect_df.to_csv(
        os.path.join(output_dir, "incorrect_examples.csv"), index=False
    )

    # Subject-wise accuracy
    subj_df = subject_accuracy(df)
    if not subj_df.empty:
        subj_path = os.path.join(output_dir, "subject_accuracy.csv")
        subj_df.to_csv(subj_path)
        print(f"Subject-wise accuracy saved to: {subj_path}")
        print("\nTop 10 subjects by accuracy:")
        print(subj_df.head(10).to_string())

    return metrics
