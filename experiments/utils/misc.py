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

def save_metrics_csv(metrics: Dict[str, Any], output_path: Path, filename: str = "metrics.csv"):
    """Save metrics to CSV file."""
    filepath = f"{output_path}/{filename}"

    # Convert defaultdict to regular dict for JSON serialization
    if isinstance(metrics, defaultdict):
        metrics = dict(metrics)

    with open(filepath, 'w', newline='') as csvfile:
        if not metrics:
            return

        # Get all unique keys from nested structure
        all_keys = set()
        for step_data in metrics.values():
            if isinstance(step_data, dict):
                all_keys.update(step_data.keys())
            else:
                all_keys.add('value')

        writer = csv.DictWriter(csvfile, fieldnames=['step'] + sorted(all_keys))
        writer.writeheader()

        for step, data in metrics.items():
            if isinstance(data, dict):
                row = {'step': step, **data}
            else:
                row = {'step': step, 'value': data}
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


def train(model, x, y, loss_fn, optimizer):
    model.train()
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


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
