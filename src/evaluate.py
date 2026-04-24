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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducible figure generation
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

    all_preds: List[int] = []
    all_labels: List[int] = []

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
    df["correct"] = df["correct"].astype(bool)
    csv_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")
    return df


def error_analysis(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return n correct and n incorrect examples for qualitative analysis."""
    correct_df = df[df["correct"]].head(n)
    incorrect_df = df[~df["correct"]].head(n)
    if len(correct_df) < n or len(incorrect_df) < n:
        print(
            f"Requested {n} examples each, but found "
            f"{len(correct_df)} correct and {len(incorrect_df)} incorrect."
        )

    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS  –  {n} Correct  |  {n} Incorrect examples")
    print("=" * 60)

    def _print_examples(subset: pd.DataFrame, tag: str) -> None:
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


def subject_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject accuracy when subject annotations are available."""
    if "subject" not in df.columns or df["subject"].nunique() <= 1:
        print("Subject information not available; skipping subject-wise accuracy.")
        return pd.DataFrame(columns=["subject", "correct_count", "total", "accuracy"])

    summary = (
        df.groupby("subject")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct_count", "count": "total"})
    )
    summary["accuracy"] = summary["correct_count"] / summary["total"]
    summary = summary.sort_values("accuracy", ascending=False).reset_index()
    return summary



def plot_error_breakdown(df: pd.DataFrame, figure_dir: str) -> str:
    """Save a bar chart of correct vs incorrect prediction counts."""
    ensure_dir(figure_dir)
    correct_count = int(df["correct"].sum())
    total_count = int(len(df))
    incorrect_count = total_count - correct_count
    categories = ["Correct", "Incorrect"]
    counts = [correct_count, incorrect_count]
    colors = ["#2ca02c", "#d62728"]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(categories, counts, color=colors)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontweight="bold")
    plt.xlabel("Outcome Category")
    plt.ylabel("Count")
    plt.title("Validation Prediction Outcome Breakdown")
    plt.tight_layout()
    plot_path = os.path.join(figure_dir, "error_breakdown.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Error breakdown figure saved to: {plot_path}")
    return plot_path


def plot_subject_accuracy(subject_df: pd.DataFrame, figure_dir: str) -> Optional[str]:
    """Save subject-wise accuracy bar chart and return the file path."""
    if subject_df.empty:
        return None
    ensure_dir(figure_dir)
    sorted_subjects = subject_df.sort_values("accuracy", ascending=False)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_subjects["subject"].astype(str), sorted_subjects["accuracy"].values)
    for bar, acc in zip(bars, sorted_subjects["accuracy"].values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.title("Subject-wise Validation Accuracy")
    plt.xlabel("Medical Subject")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plot_path = os.path.join(figure_dir, "subject_accuracy.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Subject-wise accuracy figure saved to: {plot_path}")
    return plot_path


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, figure_dir: str) -> str:
    """Save a confusion matrix figure."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    ensure_dir(figure_dir)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B", "C", "D"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix – Validation Set")
    plt.tight_layout()
    plot_path = os.path.join(figure_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {plot_path}")
    return plot_path


def plot_attention_heatmap(
    model: Any,
    dataset: Any,
    tokenizer: Any,
    figure_dir: str,
    device: Optional[torch.device] = None,
    n_examples: int = 3,
) -> None:
    """
    Save attention heatmaps for n_examples from the validation set.
    Only runs for transformer models that return attentions.
    """
    from src.lstm_model import LSTMMultipleChoice
    if isinstance(model, LSTMMultipleChoice):
        print("Skipping attention heatmap — not applicable for LSTM.")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(figure_dir)
    model.eval()
    model.to(device)

    for idx in range(min(n_examples, len(dataset))):
        example = dataset.data[idx]
        question = example["question"]
        options = [example["opa"], example["opb"], example["opc"], example["opd"]]
        true_label = int(example["cop"])

        # Tokenize the correct answer pair
        encoding = tokenizer(
            question,
            options[true_label],
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Need to reshape for multiple choice format (batch, num_choices, seq_len)
        input_ids_mc = input_ids.unsqueeze(0).expand(1, 4, -1)
        attention_mask_mc = attention_mask.unsqueeze(0).expand(1, 4, -1)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_mc,
                attention_mask=attention_mask_mc,
                output_attentions=True,
            )

        if not hasattr(outputs, "attentions") or len(outputs.attentions) == 0:
            print("Model does not return attentions — skipping heatmap.")
            return

        # Average attention across all heads in the last layer
        # attentions shape: (num_layers, batch, num_heads, seq_len, seq_len)
        last_layer_attn = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
        avg_attn = last_layer_attn[0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)

        # Get tokens for x/y axis labels
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        # Trim to non-padding tokens
        seq_len = int(attention_mask[0].sum().item())
        tokens = tokens[:seq_len]
        avg_attn = avg_attn[:seq_len, :seq_len]

        # Shorten long token lists for readability
        max_display = 30
        if len(tokens) > max_display:
            tokens = tokens[:max_display]
            avg_attn = avg_attn[:max_display, :max_display]

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(avg_attn, cmap="Blues")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(
            f"Attention Heatmap – Example {idx + 1}\n"
            f"Q: {question[:60]}...\nTrue: {LABEL_MAP[true_label]}",
            fontsize=9,
        )
        plt.tight_layout()
        plot_path = os.path.join(figure_dir, f"attention_heatmap_example_{idx + 1}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Attention heatmap saved to: {plot_path}")

def run_evaluation(
    model: Any,
    val_dataset: Any,
    output_dir: str = "outputs",
    batch_size: int = 16,
    figure_dir: str = "figures",
    device: Optional[torch.device] = None,
    tokenizer: Any = None,
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

    correct_df, incorrect_df = error_analysis(df, n=3)
    correct_df.to_csv(os.path.join(output_dir, "correct_examples.csv"), index=False)
    incorrect_df.to_csv(os.path.join(output_dir, "incorrect_examples.csv"), index=False)
    plot_error_breakdown(df, figure_dir)
    plot_confusion_matrix(preds, labels, figure_dir)
    if tokenizer is not None:
        plot_attention_heatmap(model, val_dataset, tokenizer, figure_dir)
    subj_df = subject_accuracy(df)
    subj_path = os.path.join(output_dir, "subject_accuracy.csv")
    subj_df.to_csv(subj_path, index=False)
    print(f"Subject-wise accuracy saved to: {subj_path}")
    if not subj_df.empty:
        plot_subject_accuracy(subj_df, figure_dir)

    return metrics
