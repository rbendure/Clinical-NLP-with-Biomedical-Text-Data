"""
utils.py
--------
Shared utility functions used across the pipeline.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def get_device():
    """
    Return the best available device (CUDA GPU or CPU).

    Returns
    -------
    device : torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def ensure_dir(path: str):
    """
    Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
