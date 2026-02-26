import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import math

import numpy as np
import torch
import torchvision
import torchlogix


def load_dataset(args):
    """Load a public dataset."""
    # check env varaible for dataset path
    data_path = os.getenv("DATASET_PATH", ".")
    transform = torchvision.transforms.ToTensor()
    if args.dataset == "mnist":     
        train_set = torchvision.datasets.MNIST(
            f"{data_path}/data-mnist", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            f"{data_path}/data-mnist", train=False, transform=transform
        )
    elif args.dataset == "cifar-10":
        train_set = torchvision.datasets.CIFAR10(
            f"{data_path}/data-cifar", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            f"{data_path}/data-cifar", train=False, transform=transform
        )
    
    if args.valid_set_size > 0:
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_set_size]
        )
    else:
        print(f"Training on entire training set. Using test set as validation set.")
        validation_set = test_set


    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def get_model(thresholds, args):
    """
    Select model from the architecture.
    It can be a difflogic model or a baseline model.
    """
    llkw = {
        "connections": args.connections,
        "connections_kwargs": {
            "init_method": args.connections_init_method,
            "channel_balance": args.connections_channel_balance,
            "temperature": args.connections_temperature,
            "gumbel": args.connections_gumbel
            },
        "parametrization": args.parametrization,
        "parametrization_kwargs": {
            "temperature": args.parametrization_temperature,
            "forward_sampling": args.forward_sampling,
            "weight_init": args.weight_init,
            "residual_probability": args.residual_probability,
            },
        "device": args.device,
        "lut_rank": args.lut_rank,
        "thresholds": thresholds,
        "binarization": args.binarization,
        "binarization_kwargs": {
            "one_per": args.binarization_per,
            "temperature_sampling": args.binarization_temperature,
            "temperature_softplus": args.binarization_temperature_softplus,
            "forward_sampling": args.binarization_forward_sampling
            }
    }
    model_cls = torchlogix.models.__dict__[args.architecture]
    model = model_cls(**llkw)
    return model

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
    filepath = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
                x = torchlogix.PackBitsTensor(x.reshape(x.shape[0], -1).round().bool())

            preds = model(x)

            for name, fn in eval_functions.items():
                metrics[name].append(fn(preds, y).to(torch.float32).mean().item())

    model.train(orig_mode)

    return {name: np.mean(vals) for name, vals in metrics.items()}