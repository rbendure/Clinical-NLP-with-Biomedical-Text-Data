"""
train.py
--------
Responsible: Pascual Jahuey

Handles:
  - Setting up Hugging Face Trainer with TrainingArguments
  - Running the training loop
  - Saving the trained model, tokenizer, and training logs to outputs/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments

from src.utils import ensure_dir
from src.lstm_model import LSTMMultipleChoice



# ---------------------------------------------------------------------------
# Custom data collator for multiple-choice
# ---------------------------------------------------------------------------

def mc_data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate MedMCQADataset items into batched tensors."""
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute common classification metrics for evaluation logging."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Trainer setup
# ---------------------------------------------------------------------------
# Add this class ABOVE the build_trainer function
class LSTMAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels"),
            )
        loss = outputs.loss
        logits = outputs.logits
        labels = inputs.get("labels")
        return (loss, logits, labels)
# ← LSTMAwareTrainer class ends here, no indent below

def plot_training_curves(trainer: Trainer, figure_dir: str) -> None:
    """Save train and validation loss curves from trainer log history."""
    ensure_dir(figure_dir)
    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    val_epochs, val_losses = [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            val_epochs.append(entry["epoch"])
            val_losses.append(entry["eval_loss"])
    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_losses, label="Train Loss", color="#1f77b4", alpha=0.8)
    if val_losses:
        max_step = max(train_steps) if train_steps else 1
        max_epoch = max(val_epochs) if val_epochs else 1
        val_steps = [e / max_epoch * max_step for e in val_epochs]
        plt.plot(val_steps, val_losses, label="Val Loss", color="#d62728",
                 marker="o", linewidth=2)
        for step, loss in zip(val_steps, val_losses):
            plt.annotate(f"{loss:.4f}", xy=(step, loss),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="#d62728", fontweight="bold")
    # Annotate final train loss
    if train_steps:
        plt.annotate(f"{train_losses[-1]:.4f}",
                    xy=(train_steps[-1], train_losses[-1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#1f77b4", fontweight="bold")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(figure_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {plot_path}")

def build_trainer(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str = "outputs",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Trainer:
    """Configure and return a Hugging Face Trainer for MedMCQA classification."""
    ensure_dir(output_dir)
    logs_dir = os.path.join(output_dir, "logs")
    ensure_dir(logs_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=logs_dir,
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=False,  # Disable fp16 for CPU compatibility
    )
    trainer_cls = LSTMAwareTrainer if isinstance(model, LSTMMultipleChoice) else Trainer
    return trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=mc_data_collator,
        compute_metrics=compute_metrics,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str = "outputs",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Trainer:
    """Run training and persist the final model and tokenizer."""
    trainer = build_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    print("\n=== Starting Training ===")
    trainer.train()

    model_save_path = os.path.join(output_dir, "model")
    ensure_dir(model_save_path)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to: {model_save_path}")
    plot_training_curves(trainer, output_dir)  # ← add this line
    return trainer

