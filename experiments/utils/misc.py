"""Utilities for experiment scripts."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import torch
import numpy as np

from torchlogix import PackBitsTensor
from torchlogix.models.baseline_nn import FullyConnectedNN
from torchlogix.models.conv import CNN
from torchlogix.models.nn import RandomlyConnectedNN
from torchlogix.layers.binarization import LearnableBinarization

from .shared_config import IMPL_TO_DEVICE, BITS_TO_TORCH_FLOATING_POINT_TYPE


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist, including parent directories.
        The action might throw OSError, along with other kinds of exception
        """
        # Create all parent directories if they don't exist
        os.makedirs(folder_name, exist_ok=True)

        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


# Shared experiment utilities
def save_metrics_csv(step: int, metrics: Dict[str, Any], output_path: Path, filename: str = "metrics.csv"):
    """Append single step metrics to CSV file."""
    filepath = f"{output_path}/{filename}"
    
    # Determine if file exists to write headers
    file_exists = Path(filepath).exists()

    # Convert tensor/numpy values to Python primitives
    row = {'step': step}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # numpy/torch scalar
            row[key] = value.item()
        else:
            row[key] = value
    
    fieldnames = ['step'] + sorted(metrics.keys())
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_thresholds_csv(
    step: int, 
    thresholds: torch.Tensor, 
    output_path: Path, 
    filename: str = "thresholds.csv"
):
    """Append threshold values to CSV file.
    
    Args:
        step: Training step number
        thresholds: Threshold tensor with last dimension = num_bits
                   Shapes can be (num_bits,), (num_features, num_bits), 
                   (num_channels, num_bits), or higher dimensional
        output_path: Directory to save CSV
        filename: Name of CSV file
    """
    filepath = f"{output_path}/{filename}"
    file_exists = Path(filepath).exists()

    # Flatten thresholds into a single row
    row = {'step': step}
    
    # Convert to numpy for easier indexing
    thresh_np = thresholds.detach().cpu().numpy()
    
    if thresh_np.ndim == 1:
        # Global case: (num_bits,)
        for bit_idx in range(len(thresh_np)):
            col_name = f"thresh_{bit_idx}"
            row[col_name] = float(thresh_np[bit_idx])
    
    else:
        # Multi-dimensional case: (..., num_bits)
        # Iterate through all indices
        import numpy as np
        for index in np.ndindex(thresh_np.shape):
            # Create column name from indices: thresh_0_1_2 for index (0,1,2)
            col_name = "thresh_" + "_".join(map(str, index))
            row[col_name] = float(thresh_np[index])
    
    # Write to CSV
    fieldnames = ['step'] + sorted([k for k in row.keys() if k != 'step'])
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



def save_config(config: Dict[str, Any], output_path: Path, filename: str = "config.json"):
    """Save configuration to JSON file."""
    filepath = f"{output_path}/{filename}"

    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        elif hasattr(value, '__dict__'):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2, default=str)


def load_model_from_checkpoint(model_path: Path, model_class, **model_kwargs):
    """Load a trained model from checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize model
    model = model_class(**model_kwargs)

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    return model


def train(model, x, y, loss_fn, optimizer, regularization_method, regularization_weight):
    model.train()
    x = model(x)
    loss = loss_fn(x, y)
    # Regularization
    if regularization_weight > 0.0:
        reg_loss = 0.0
        for layer in model:
            if hasattr(layer, 'get_regularization_loss'):
                reg_loss += layer.get_regularization_loss(regularization_method)
        loss += regularization_weight * reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # return loss.item()
    return loss


def evaluate_model(model, loader, eval_functions, mode="eval", device="cuda"):
    """Evaluate model on a data loader with given evaluation functions.
    Assumes metrics can be computed in batches and averaged."""
    orig_mode = model.training
    model.train(mode == "train")

    metrics = defaultdict(list)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if mode == "packbits":
                x = PackBitsTensor(x.reshape(x.shape[0], -1).round().bool())

            preds = model(x)

            for name, fn in eval_functions.items():
                metrics[name].append(fn(preds, y).to(torch.float32).mean().item())

    model.train(orig_mode)

    return {name: np.mean(vals) for name, vals in metrics.items()}


def create_eval_functions(loss_fn):
    """Create standard evaluation functions."""
    return {
        "loss": loss_fn,
        "acc": lambda preds, y: (preds.argmax(-1) == y).to(torch.float32).mean(),
    }


def print_memory_usage(stage_name):
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"{stage_name:30s} | Allocated: {allocated:6.2f} GB | Reserved: {reserved:6.2f} GB")    
