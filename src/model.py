"""
model.py
--------
Responsible: Pascual Jahuey

Handles:
  - Tokenizer initialisation
  - Multiple-choice classification model initialisation
  - Support for switching between DistilBERT and BERT
"""

from typing import Dict

from transformers import AutoModelForMultipleChoice, AutoTokenizer

# Supported model identifiers
SUPPORTED_MODELS: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "bert-base-uncased": "bert-base-uncased",
}

DEFAULT_MODEL = "distilbert-base-uncased"


def resolve_model_name(model_name: str) -> str:
    """Resolve shorthand names and validate supported model identifiers."""
    normalized = model_name.strip().lower()
    if normalized not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(set(SUPPORTED_MODELS.values())))
        alias_keys = [key for key, value in SUPPORTED_MODELS.items() if key != value]
        aliases = ", ".join(
            sorted(alias_keys)
        )
        raise ValueError(
            "Invalid model name: "
            f"'{model_name}'. Supported models are: {supported}. "
            f"You may also use aliases: {aliases}."
        )
    return SUPPORTED_MODELS[normalized]


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load and return a Hugging Face tokenizer."""
    resolved_model = resolve_model_name(model_name)
    print(f"Loading tokenizer: {resolved_model}")
    return AutoTokenizer.from_pretrained(resolved_model)


def get_model(model_name: str = DEFAULT_MODEL, num_labels: int = 4):
    """
    Load a pretrained transformer model adapted for multiple-choice
    classification (4-way: A/B/C/D).
    """
    resolved_model = resolve_model_name(model_name)
    print(f"Loading model: {resolved_model}  (num_choices={num_labels})")
    return AutoModelForMultipleChoice.from_pretrained(resolved_model)
