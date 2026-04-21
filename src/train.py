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
import torch
from transformers import Trainer, TrainingArguments

from src.utils import ensure_dir


# ---------------------------------------------------------------------------
# Custom data collator for multiple-choice
# ---------------------------------------------------------------------------

def mc_data_collator(features):
    """
    Collate a list of MedMCQADataset items into batched tensors.

    Each item has:
        input_ids      : (4, seq_len)
        attention_mask : (4, seq_len)
        labels         : int scalar

    After collation:
        input_ids      : (batch, 4, seq_len)
        attention_mask : (batch, 4, seq_len)
        labels         : (batch,)
    """
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Trainer setup
# ---------------------------------------------------------------------------

def build_trainer(
    model,
    train_dataset,
    val_dataset,
    output_dir: str = "outputs",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
):
    """
    Configure and return a Hugging Face Trainer for multiple-choice
    classification.

    Parameters
    ----------
    model          : Pretrained multiple-choice model
    train_dataset  : MedMCQADataset (training split)
    val_dataset    : MedMCQADataset (validation split)
    output_dir     : Root directory for checkpoints and logs
    num_epochs     : Number of training epochs (recommended 2-3)
    batch_size     : Per-device batch size
    learning_rate  : AdamW learning rate
    seed           : Random seed for Trainer

    Returns
    -------
    trainer : Hugging Face Trainer
    """
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
        metric_for_best_model="eval_loss",
        logging_dir=logs_dir,
        logging_steps=50,
        seed=seed,
        report_to="none",           # disable wandb / external trackers
        fp16=torch.cuda.is_available(),  # use mixed precision on GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=mc_data_collator,
    )
    return trainer


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str = "outputs",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
):
    """
    Run the complete training loop and save the final model + tokenizer.

    Parameters
    ----------
    model, tokenizer : Loaded model and tokenizer
    train_dataset    : MedMCQADataset (training split)
    val_dataset      : MedMCQADataset (validation split)
    output_dir       : Directory to save model, tokenizer, and logs
    num_epochs       : Training epochs
    batch_size       : Per-device batch size
    learning_rate    : Learning rate
    seed             : Random seed

    Returns
    -------
    trainer : Fitted Trainer object (use for evaluation)
    """
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

    # Persist model and tokenizer
    model_save_path = os.path.join(output_dir, "model")
    ensure_dir(model_save_path)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to: {model_save_path}")

    return trainer
