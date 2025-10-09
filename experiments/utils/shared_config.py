"""Shared configuration and constants for experiments."""

import torch
from pathlib import Path

# Type mappings
BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}

IMPL_TO_DEVICE = {"cuda": "cuda", "python": "cpu"}

# Dataset choices
DATASET_CHOICES = [
    "adult",
    "breast_cancer",
    "monk1",
    "monk2",
    "monk3",
    "mnist",
    "mnist20x20",
    "cifar-10-3-thresholds",
    "cifar-10-31-thresholds",
    "cora",
    "pubmed",
    "citeseer",
    "nell",
]

ARCHITECTURE_CHOICES = [
    "randomly_connected",
    "fully_connected",
    "cnn",
    "tiny",
    "small",
    "paper",
]

def setup_experiment(seed: int, implementation: str):
    """Setup experiment environment with reproducible settings."""
    import random
    import numpy as np

    # Set random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set device
    device = IMPL_TO_DEVICE[implementation]

    # Optimization settings
    torch.set_num_threads(1)

    return device