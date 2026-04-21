"""
model.py
--------
Responsible: Pascual Jahuey

Handles:
  - Tokenizer initialisation
  - Multiple-choice classification model initialisation
  - Support for switching between DistilBERT and BERT
"""

from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
)

# Supported model identifiers
SUPPORTED_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
}

DEFAULT_MODEL = "distilbert-base-uncased"


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """
    Load and return a Hugging Face tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or shorthand key from SUPPORTED_MODELS.

    Returns
    -------
    tokenizer : AutoTokenizer
    """
    # Allow shorthand keys like "distilbert" or "bert"
    model_name = SUPPORTED_MODELS.get(model_name, model_name)
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_model(model_name: str = DEFAULT_MODEL, num_labels: int = 4):
    """
    Load a pretrained transformer model adapted for multiple-choice
    classification (4-way: A/B/C/D).

    The AutoModelForMultipleChoice head takes four candidate encodings and
    outputs a single logit per candidate, yielding a 4-class distribution
    after softmax.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or shorthand key from SUPPORTED_MODELS.
    num_labels : int
        Number of answer choices (fixed at 4 for MedMCQA).

    Returns
    -------
    model : AutoModelForMultipleChoice
    """
    model_name = SUPPORTED_MODELS.get(model_name, model_name)
    print(f"Loading model: {model_name}  (num_choices={num_labels})")
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    return model
