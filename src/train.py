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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments

from src.utils import ensure_dir


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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=logs_dir,
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    return Trainer(
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

    return trainer
