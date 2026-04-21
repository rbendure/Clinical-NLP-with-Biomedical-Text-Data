"""
data.py
-------
Responsible: James Garner

Handles:
  - Loading the MedMCQA dataset from Hugging Face
  - Subsetting to configurable sizes for speed
  - Preprocessing each example into (question + option) pairs
  - Tokenizing inputs and encoding labels for 4-class classification
"""

from typing import Any, Dict, List, Tuple

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
NUM_LABELS = 4


def _validate_subset_sizes(train_size: int, val_size: int, train_total: int, val_total: int) -> None:
    """Validate user-provided subset sizes with clear, beginner-friendly errors."""
    if train_size <= 0:
        raise ValueError(f"train_size must be > 0. Received: {train_size}")
    if val_size <= 0:
        raise ValueError(f"val_size must be > 0. Received: {val_size}")
    if train_size > train_total:
        raise ValueError(
            f"train_size ({train_size}) is larger than available training examples ({train_total})."
        )
    if val_size > val_total:
        raise ValueError(
            f"val_size ({val_size}) is larger than available validation examples ({val_total})."
        )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_medmcqa(train_size: int = 5000, val_size: int = 1000) -> Tuple[Dataset, Dataset]:
    """Load MedMCQA and return deterministic train/validation subsets."""
    print(f"Loading MedMCQA dataset (train={train_size}, val={val_size})...")
    dataset = load_dataset("openlifescienceai/medmcqa")

    train_total = len(dataset["train"])
    val_total = len(dataset["validation"])
    _validate_subset_sizes(train_size, val_size, train_total, val_total)

    train_data = dataset["train"].select(range(train_size))
    val_data = dataset["validation"].select(range(val_size))

    print(f"  Training samples  : {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _format_input(example: Dict[str, Any]) -> Tuple[str, List[str], int]:
    """Build four (question + option) strings from one MedMCQA example."""
    question = example["question"]
    options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    label = int(example["cop"])  # 0 → A, 1 → B, 2 → C, 3 → D
    return question, options, label


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MedMCQADataset(TorchDataset):
    """PyTorch-compatible dataset for MedMCQA multiple-choice classification."""

    def __init__(self, hf_dataset: Dataset, tokenizer: Any, max_length: int = 128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        question, options, label = _format_input(example)

        # Tokenize all four (question, option) pairs
        encodings = self.tokenizer(
            [question] * NUM_LABELS,
            options,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": label,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_datasets(
    tokenizer: Any,
    train_size: int = 5000,
    val_size: int = 1000,
    max_length: int = 128,
) -> Tuple[MedMCQADataset, MedMCQADataset]:
    """Full pipeline: load → subset → wrap in MedMCQADataset."""
    train_raw, val_raw = load_medmcqa(train_size, val_size)
    train_dataset = MedMCQADataset(train_raw, tokenizer, max_length)
    val_dataset = MedMCQADataset(val_raw, tokenizer, max_length)
    return train_dataset, val_dataset
