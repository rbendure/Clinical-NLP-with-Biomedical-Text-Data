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
"""

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducible figure generation
import matplotlib.pyplot as plt
import pandas as pd

from src.data import build_datasets
from src.evaluate import run_evaluation
from src.model import DEFAULT_MODEL, get_model, get_tokenizer, resolve_model_name
from src.train import run_training
from src.utils import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Biomedical NLP – Multiple-Choice Text Classification on MedMCQA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Single model name/alias (distilbert, bert, distilbert-base-uncased, "
            "bert-base-uncased) or comma-separated pair for comparison"
        ),
    )
    parser.add_argument(
        "--compare_models",
        action="store_true",
        help="Run both distilbert-base-uncased and bert-base-uncased for direct comparison",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--train_size", type=int, default=5000, help="Training subset size")
    parser.add_argument("--val_size", type=int, default=1000, help="Validation subset size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model outputs and artifacts",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="figures",
        help="Directory to save generated figures",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_and_validate_models(model_arg: str, compare_models: bool) -> List[str]:
    """Parse, resolve, and validate the final list of models to run."""
    if compare_models:
        return ["distilbert-base-uncased", "bert-base-uncased"]

    requested = [part.strip() for part in model_arg.split(",") if part.strip()]
    if not requested:
        raise ValueError("No model specified. Provide --model with a valid identifier.")

    resolved = [resolve_model_name(name) for name in requested]

    # Remove duplicates while preserving order
    unique_resolved: List[str] = []
    for model_name in resolved:
        if model_name not in unique_resolved:
            unique_resolved.append(model_name)

    return unique_resolved


def save_config(config: Dict[str, Any], output_dir: str) -> str:
    """Save experiment config to outputs/config.json."""
    ensure_dir(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


def save_model_comparison(results: List[Dict[str, Any]], output_dir: str, figure_dir: str) -> None:
    """Save model comparison CSV, README placeholders, and comparison figure."""
    if len(results) < 1:
        return

    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    comparison_df = pd.DataFrame(results)
    comparison_csv = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Model comparison saved to: {comparison_csv}")

    # Placeholder map for README replacement workflows
    placeholder_payload = {
        "{{DISTILBERT_VAL_ACCURACY}}": None,
        "{{BERT_VAL_ACCURACY}}": None,
        "{{BEST_MODEL}}": None,
    }
    model_placeholder_map = {
        "distilbert-base-uncased": "{{DISTILBERT_VAL_ACCURACY}}",
        "bert-base-uncased": "{{BERT_VAL_ACCURACY}}",
    }
    for row in results:
        placeholder_key = model_placeholder_map.get(row["model"])
        if placeholder_key is not None:
            placeholder_payload[placeholder_key] = round(row["accuracy"], 4)

    best_model = max(results, key=lambda r: r["accuracy"])["model"]
    placeholder_payload["{{BEST_MODEL}}"] = best_model

    placeholders_path = os.path.join(output_dir, "readme_placeholders.json")
    with open(placeholders_path, "w", encoding="utf-8") as f:
        json.dump(placeholder_payload, f, indent=2)
    print(f"README placeholders saved to: {placeholders_path}")

    plot_df = comparison_df[["model", "accuracy"]].copy()
    plot_df = pd.concat(
        [
            plot_df,
            pd.DataFrame([{"model": "random_baseline_25%", "accuracy": 0.25}]),
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#7f7f7f"]
    plt.bar(plot_df["model"], plot_df["accuracy"], color=colors[: len(plot_df)])
    plt.ylim(0, 1)
    plt.xlabel("Model Name")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison: Validation Accuracy")
    plt.tight_layout()

    plot_path = os.path.join(figure_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Model comparison figure saved to: {plot_path}")


def run_single_model(
    model_name: str, args: argparse.Namespace, device: Any, multi_run: bool
) -> Dict[str, Any]:
    """Run train+evaluate for one model and return summary metrics."""
    model_output_dir = args.output_dir if not multi_run else os.path.join(args.output_dir, model_name)
    model_figure_dir = args.figure_dir if not multi_run else os.path.join(args.figure_dir, model_name)

    ensure_dir(model_output_dir)
    ensure_dir(model_figure_dir)

    model_config = {
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "output_dir": model_output_dir,
        "figure_dir": model_figure_dir,
    }
    config_path = save_config(model_config, model_output_dir)
    print(f"Run configuration saved to: {config_path}")

    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    train_dataset, val_dataset = build_datasets(
        tokenizer=tokenizer,
        train_size=args.train_size,
        val_size=args.val_size,
        max_length=args.max_length,
    )

    trainer = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=model_output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    metrics = run_evaluation(
        model=trainer.model,
        val_dataset=val_dataset,
        output_dir=model_output_dir,
        batch_size=args.eval_batch_size,
        figure_dir=model_figure_dir,
        device=device,
    )

    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "output_dir": model_output_dir,
    }


def main() -> None:
    """Main entry point for MedMCQA text classification experiments."""
    args = parse_args()

    try:
        models_to_run = resolve_and_validate_models(args.model, args.compare_models)
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    print("\n" + "=" * 70)
    print("  Biomedical NLP – MedMCQA Text Classification Pipeline")
    print("=" * 70)
    print(f"  Models      : {', '.join(models_to_run)}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Train size  : {args.train_size}")
    print(f"  Val size    : {args.val_size}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Figure dir  : {args.figure_dir}")
    print("=" * 70 + "\n")

    set_seed(args.seed)
    device = get_device()

    ensure_dir(args.output_dir)
    ensure_dir(args.figure_dir)

    overall_config = {
        "models": models_to_run,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "figure_dir": args.figure_dir,
    }
    root_config_path = save_config(overall_config, args.output_dir)
    print(f"Global configuration saved to: {root_config_path}")

    all_results: List[Dict[str, Any]] = []
    for model_name in models_to_run:
        print(f"\n>>> Running experiment for model: {model_name}")
        try:
            result = run_single_model(model_name, args, device, multi_run=len(models_to_run) > 1)
        except ValueError as exc:
            raise SystemExit(f"Configuration error: {exc}") from exc
        all_results.append(result)

    save_model_comparison(all_results, args.output_dir, args.figure_dir)

    print("\n" + "=" * 70)
    for row in all_results:
        print(f"  {row['model']}: accuracy={row['accuracy'] * 100:.2f}%")
    print("=" * 70)
    print(f"\nAll outputs saved under: {os.path.abspath(args.output_dir)}")
    print(f"All figures saved under: {os.path.abspath(args.figure_dir)}")


if __name__ == "__main__":
    main()
