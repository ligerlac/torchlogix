"""Utilities package for TorchLogix experiments."""

# Import and re-export all utility functions and classes
from .misc import (
    CreateFolder, IsReadableDir, IsValidFile,
    save_metrics_csv, save_config, load_model_from_checkpoint,
    evaluate_model, create_eval_functions, train
)
from .loading import load_dataset, load_n
from .model_selection import get_model
from .shared_config import (
    DATASET_CHOICES, ARCHITECTURE_CHOICES, BITS_TO_TORCH_FLOATING_POINT_TYPE,
    IMPL_TO_DEVICE, setup_experiment
)
from .drawing import plot_loss_histories


__all__ = [
    # argparse actions
    "CreateFolder", "IsReadableDir", "IsValidFile",
    # file operations
    "save_metrics_csv", "save_config", "load_model_from_checkpoint",
    # model operations
    "evaluate_model", "create_eval_functions", "train",
    # data loading
    "load_dataset", "load_n",
    # model creation
    "get_model",
    # configuration
    "DATASET_CHOICES", "ARCHITECTURE_CHOICES", "BITS_TO_TORCH_FLOATING_POINT_TYPE",
    "IMPL_TO_DEVICE", "setup_experiment",
    # plotting
    "plot_loss_curves",
]