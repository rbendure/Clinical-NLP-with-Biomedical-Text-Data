"""
main.py
-------
Responsible: Pascual Jahuey (full integration)

End-to-end pipeline:
  1. Set seed and device
  2. Load tokenizer and model
  3. Build train/val datasets
  4. Train the model
  5. Evaluate and save results

Usage:
    python -m src.main [--model distilbert] [--epochs 3] [--batch_size 8]
                       [--train_size 5000] [--val_size 1000]
                       [--output_dir outputs]
"""

import argparse
import os

from src.utils import set_seed, get_device
from src.model import get_tokenizer, get_model
from src.data import build_datasets
from src.train import run_training
from src.evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Biomedical NLP – Multiple-Choice Question Answering Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name or shorthand key (distilbert / bert)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Per-device batch size (default 8)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="AdamW learning rate (default 2e-5)"
    )
    parser.add_argument(
        "--train_size", type=int, default=5000,
        help="Number of training samples to use (default 5000)"
    )
    parser.add_argument(
        "--val_size", type=int, default=1000,
        help="Number of validation samples to use (default 1000)"
    )
    parser.add_argument(
        "--max_length", type=int, default=128,
        help="Max token length per (question, option) pair (default 128)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save model, logs, and results (default outputs/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Biomedical NLP – MedMCQA Classification Pipeline")
    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Train size  : {args.train_size}")
    print(f"  Val size    : {args.val_size}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 60 + "\n")

    # 1. Reproducibility
    set_seed(args.seed)
    device = get_device()

    # 2. Tokenizer and model
    tokenizer = get_tokenizer(args.model)
    model = get_model(args.model)

    # 3. Datasets
    train_dataset, val_dataset = build_datasets(
        tokenizer=tokenizer,
        train_size=args.train_size,
        val_size=args.val_size,
        max_length=args.max_length,
    )

    # 4. Training
    trainer = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    # 5. Evaluation
    metrics = run_evaluation(
        model=trainer.model,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        device=device,
    )

    print("\n" + "=" * 60)
    print(f"  Final Validation Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print("=" * 60)
    print(f"\nAll outputs saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
