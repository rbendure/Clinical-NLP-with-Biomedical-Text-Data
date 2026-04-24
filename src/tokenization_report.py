"""
tokenization_report.py
----------------------
Responsible: Riley Bendure
Handles:
  - Documenting tokenization behavior for both LSTM and transformer models
  - Generating vocabulary statistics and unknown token analysis
  - Saving tokenization report to outputs/
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from src.utils import ensure_dir

TOKENIZER_NAME = "bert-base-uncased"


def analyze_tokenization(train_data: Dataset, tokenizer, output_dir: str, figure_dir: str) -> dict:
    """Analyze tokenization behavior and return summary statistics."""
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    questions = [ex["question"] for ex in train_data]
    options = [[ex["opa"], ex["opb"], ex["opc"], ex["opd"]] for ex in train_data]

    # Tokenize all questions
    token_lengths = []
    truncated_count = 0
    unk_count = 0
    total_tokens = 0

    for question, opts in zip(questions, options):
        for opt in opts:
            encoded = tokenizer(
                question, opt,
                max_length=128,
                truncation=True,
                padding="max_length",
            )
            input_ids = encoded["input_ids"]
            # Count truncated sequences (no padding means it was truncated)
            actual_length = sum(1 for x in input_ids if x != tokenizer.pad_token_id)
            token_lengths.append(actual_length)
            if actual_length == 128:
                truncated_count += 1
            # Count UNK tokens
            unk_count += input_ids.count(tokenizer.unk_token_id)
            total_tokens += actual_length

    total_sequences = len(token_lengths)
    truncated_pct = truncated_count / total_sequences * 100

    report = {
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_type": "WordPiece (shared for both LSTM and Transformer models)",
        "vocab_size": tokenizer.vocab_size,
        "max_length": 128,
        "padding_strategy": "max_length",
        "truncation": True,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "total_sequences_analyzed": total_sequences,
        "truncated_sequences": truncated_count,
        "truncated_pct": round(truncated_pct, 2),
        "total_unk_tokens": unk_count,
        "avg_token_length": round(float(np.mean(token_lengths)), 2),
        "max_token_length": int(np.max(token_lengths)),
        "min_token_length": int(np.min(token_lengths)),
        "lstm_notes": {
            "embedding": "Learned from scratch — nn.Embedding(vocab_size=30522, embed_dim=128)",
            "unknown_tokens": "Handled by BERT tokenizer [UNK] token (id=100)",
            "vocabulary": "Reuses BERT WordPiece vocabulary (30,522 tokens)",
            "padding": "Packed sequences used in LSTM to ignore padding during forward pass",
        },
        "transformer_notes": {
            "embedding": "Pretrained contextual embeddings from bert-base-uncased / distilbert-base-uncased",
            "unknown_tokens": "Handled by BERT tokenizer [UNK] token (id=100)",
            "vocabulary": "WordPiece vocabulary (30,522 tokens)",
            "padding": "Attention mask used to ignore padding tokens",
        },
    }

    # Save report
    report_path = os.path.join(output_dir, "tokenization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Tokenization report saved to: {report_path}")

    return report, token_lengths


def plot_token_length_distribution(token_lengths: list, figure_dir: str) -> None:
    """Plot distribution of token lengths after tokenization."""
    ensure_dir(figure_dir)
    plt.figure(figsize=(9, 5))
    plt.hist(token_lengths, bins=40, color="#1f77b4", edgecolor="white")
    mean_len = np.mean(token_lengths)
    plt.axvline(mean_len, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_len:.1f}")
    plt.axvline(128, color="black", linestyle="--", linewidth=1.5,
                label="Max length: 128")
    plt.xlabel("Token Length")
    plt.ylabel("Count")
    plt.title("Token Length Distribution (Question + Option pairs)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(figure_dir, "token_length_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Token length distribution saved to: {plot_path}")


def plot_tokenization_comparison(figure_dir: str) -> None:
    """Plot a side-by-side comparison of LSTM vs Transformer tokenization approach."""
    ensure_dir(figure_dir)
    categories = ["Vocab Size", "Embedding Dim", "Max Length", "UNK Handling"]
    lstm_vals = [30522, 128, 128, 1]
    transformer_vals = [30522, 768, 128, 1]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, lstm_vals, width, label="LSTM", color="#ff7f0e")
    bars2 = ax.bar(x + width/2, transformer_vals, width, label="Transformer (BERT)", color="#1f77b4")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Value")
    ax.set_title("Tokenization: LSTM vs Transformer Comparison")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(figure_dir, "tokenization_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Tokenization comparison figure saved to: {plot_path}")


def run_tokenization_report(train_data: Dataset, output_dir: str, figure_dir: str) -> None:
    """Run full tokenization analysis and save all artifacts."""
    print("\n=== Running Tokenization Analysis ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    report, token_lengths = analyze_tokenization(train_data, tokenizer, output_dir, figure_dir)
    plot_token_length_distribution(token_lengths, figure_dir)
    plot_tokenization_comparison(figure_dir)
    print(f"  Vocab size          : {report['vocab_size']}")
    print(f"  Avg token length    : {report['avg_token_length']}")
    print(f"  Truncated sequences : {report['truncated_sequences']} ({report['truncated_pct']}%)")
    print(f"  Total UNK tokens    : {report['total_unk_tokens']}")
    print("=== Tokenization Analysis Complete ===")
    