"""
eda.py
------
Responsible: Riley Bendure
Handles:
  - Exploratory Data Analysis on MedMCQA dataset
  - Label distribution, question length, subject distribution, vocabulary stats
  - Saves all figures to figures/eda/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import Dataset
from src.utils import ensure_dir


def plot_label_distribution(train_data: Dataset, figure_dir: str) -> None:
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    labels = [label_map[int(ex["cop"])] for ex in train_data]
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values, color="#1f77b4")
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontweight="bold")
    plt.xlabel("Answer Choice")
    plt.ylabel("Count")
    plt.title("Label Distribution (Training Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "label_distribution.png"), dpi=200)
    plt.close()
    print("  Saved: label_distribution.png")


def plot_subject_distribution(train_data: Dataset, figure_dir: str) -> None:
    subjects = [ex.get("subject_name", "unknown") for ex in train_data]
    counts = pd.Series(subjects).value_counts().head(15)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(counts.index, counts.values, color="#2ca02c")
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Subject")
    plt.ylabel("Count")
    plt.title("Top 15 Subjects by Frequency (Training Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "subject_distribution.png"), dpi=200)
    plt.close()
    print("  Saved: subject_distribution.png")

def plot_question_length(train_data: Dataset, figure_dir: str) -> None:
    """Plot and save histogram of question lengths in words."""
    lengths = [len(ex["question"].split()) for ex in train_data]
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=40, color="#ff7f0e", edgecolor="white")
    plt.xlabel("Question Length (words)")
    plt.ylabel("Count")
    plt.title("Question Length Distribution (Training Set)")
    # Annotate mean and max
    plt.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {np.mean(lengths):.1f}")
    plt.axvline(np.max(lengths), color="black", linestyle="--", linewidth=1.5,
                label=f"Max: {np.max(lengths)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "question_length.png"), dpi=200)
    plt.close()
    print("  Saved: question_length.png")


def plot_option_length(train_data: Dataset, figure_dir: str) -> None:
    option_keys = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    avg_lengths = {
        label: np.mean([len(ex[key].split()) for ex in train_data])
        for label, key in option_keys.items()
    }
    plt.figure(figsize=(7, 5))
    bars = plt.bar(avg_lengths.keys(), avg_lengths.values(), color="#9467bd")
    for bar, val in zip(bars, avg_lengths.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    plt.xlabel("Answer Choice")
    plt.ylabel("Average Length (words)")
    plt.title("Average Option Length by Answer Choice (Training Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "option_length.png"), dpi=200)
    plt.close()
    print("  Saved: option_length.png")

def print_summary_stats(train_data: Dataset, val_data: Dataset) -> None:
    """Print basic dataset statistics to the console."""
    question_lengths = [len(ex["question"].split()) for ex in train_data]
    print("\n=== EDA Summary Statistics ===")
    print(f"  Training samples       : {len(train_data)}")
    print(f"  Validation samples     : {len(val_data)}")
    print(f"  Unique subjects        : {len(set(ex.get('subject_name', '') for ex in train_data))}")
    print(f"  Avg question length    : {np.mean(question_lengths):.1f} words")
    print(f"  Max question length    : {max(question_lengths)} words")
    print(f"  Min question length    : {min(question_lengths)} words")
    print("=" * 40)


def run_eda(train_data: Dataset, val_data: Dataset, figure_dir: str) -> None:
    """Run full EDA pipeline and save all figures."""
    eda_figure_dir = os.path.join(figure_dir, "eda")
    ensure_dir(eda_figure_dir)
    print("\n=== Running EDA ===")
    print_summary_stats(train_data, val_data)
    plot_label_distribution(train_data, eda_figure_dir)
    plot_question_length(train_data, eda_figure_dir)
    plot_subject_distribution(train_data, eda_figure_dir)
    plot_option_length(train_data, eda_figure_dir)
    print(f"EDA figures saved to: {eda_figure_dir}")

     