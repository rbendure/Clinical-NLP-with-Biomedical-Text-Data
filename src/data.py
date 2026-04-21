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

from datasets import load_dataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
NUM_LABELS = 4


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_medmcqa(train_size: int = 5000, val_size: int = 1000):
    """
    Load the MedMCQA dataset from Hugging Face and return subsets.

    Parameters
    ----------
    train_size : int
        Number of training examples to keep (default 5000).
    val_size : int
        Number of validation examples to keep (default 1000).

    Returns
    -------
    train_data, val_data : HuggingFace Dataset slices
    """
    print(f"Loading MedMCQA dataset (train={train_size}, val={val_size})...")
    dataset = load_dataset("openlifescienceai/medmcqa")

    train_data = dataset["train"].select(range(train_size))
    val_data = dataset["validation"].select(range(val_size))

    print(f"  Training samples  : {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _format_input(example):
    """
    Build four (question + option) strings from a single MedMCQA example.

    MedMCQA fields used:
      question, opa, opb, opc, opd  – option text
      cop                            – correct option index (0-based int)

    Returns a dict with a single key 'label' (int 0-3) and four candidate
    strings.  The actual tokenisation is done in MedMCQADataset.
    """
    question = example["question"]
    options = [
        example["opa"],
        example["opb"],
        example["opc"],
        example["opd"],
    ]
    label = int(example["cop"])  # 0 → A, 1 → B, 2 → C, 3 → D
    return question, options, label


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MedMCQADataset(Dataset):
    """
    PyTorch-compatible Dataset that tokenizes MedMCQA examples on the fly.

    Each example is represented as four (question, option_i) token pairs.
    The model receives ALL four pairs concatenated under a single label.
    We use the standard multiple-choice encoding:
        input_ids shape: (4, seq_len)  per example.
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 128):
        """
        Parameters
        ----------
        hf_dataset  : Hugging Face Dataset (already sliced)
        tokenizer   : Hugging Face tokenizer
        max_length  : Maximum token length per (question, option) pair
        """
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question, options, label = _format_input(example)

        # Tokenize all four (question, option) pairs
        encodings = self.tokenizer(
            [question] * NUM_LABELS,          # repeated question
            options,                           # four options
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],           # (4, max_length)
            "attention_mask": encodings["attention_mask"], # (4, max_length)
            "labels": label,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_datasets(tokenizer, train_size: int = 5000, val_size: int = 1000,
                   max_length: int = 128):
    """
    Full pipeline: load → subset → wrap in MedMCQADataset.

    Parameters
    ----------
    tokenizer  : Hugging Face tokenizer
    train_size : Number of training examples
    val_size   : Number of validation examples
    max_length : Maximum token length per candidate pair

    Returns
    -------
    train_dataset, val_dataset : MedMCQADataset instances
    """
    train_raw, val_raw = load_medmcqa(train_size, val_size)
    train_dataset = MedMCQADataset(train_raw, tokenizer, max_length)
    val_dataset = MedMCQADataset(val_raw, tokenizer, max_length)
    return train_dataset, val_dataset
