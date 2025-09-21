#!/usr/bin/env python3
"""Training script for TorchLogix models."""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from shared_config import (
    DATASET_CHOICES, ARCHITECTURE_CHOICES, BITS_TO_TORCH_FLOATING_POINT_TYPE,
    IMPL_TO_DEVICE, setup_experiment
)
from utils import CreateFolder, save_metrics_csv, save_config, create_eval_functions, evaluate_model
from model_selection import get_model
from loading import load_dataset, load_n
from torchlogix.utils import train


def run_training(args):
    """Run the training loop."""
    # Setup experiment
    device = setup_experiment(args.seed, args.implementation)

    # Load data
    train_loader, validation_loader, test_loader = load_dataset(args)

    # Get model, loss, and optimizer
    model, loss_fn, optim = get_model(args)
    model.to(device)

    # Create evaluation functions
    eval_functions = create_eval_functions(loss_fn)

    # Training tracking
    metrics = defaultdict(list)
    best_val_acc = 0.0

    print(f"Starting training for {args.num_iterations} iterations...")
    print(f"Model: {args.architecture}, Dataset: {args.dataset}")
    print(f"Device: {device}, Implementation: {args.implementation}")

    # Training loop
    for i, (x, y) in tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="Training",
        total=args.num_iterations,
    ):
        # Move data to device
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)
        y = y.to(device)

        # Training step
        loss = train(model, x, y, loss_fn, optim)
        print(f"loss = {loss}")

        # Log training loss
        metrics[i + 1] = {"train_loss": loss}

        # Evaluation
        if (i + 1) % args.eval_freq == 0:
            print(f"\nEvaluation at iteration {i + 1}")

            # Evaluate on validation set
            eval_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="eval", device=device
            )
            train_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="train", device=device
            )

            # Update metrics
            metrics[i + 1].update(
                {f"val_{k}_eval": v for k, v in eval_metrics.items()} |
                {f"val_{k}_train": v for k, v in train_metrics.items()}
            )

            # Check for best model
            if eval_metrics["acc"] > best_val_acc:
                best_val_acc = eval_metrics["acc"]
                print(f"New best validation accuracy: {best_val_acc:.4f}")

                # Save best model
                torch.save(model.state_dict(), f"{args.output}/best_model.pt")

            print(f"Validation - Loss (train mode): {train_metrics['loss']:.4f}, Acc (train mode): {train_metrics['acc']:.4f}, Loss (eval mode): {eval_metrics['loss']:.4f}, Acc (eval mode): {eval_metrics['acc']:.4f}")

            # Save intermediate metrics
            save_metrics_csv(metrics, args.output, "training_metrics.csv")

    # Save final model
    torch.save(model.state_dict(), f"{args.output}/final_model.pt")

    # Save final metrics and config
    save_metrics_csv(metrics, args.output, "training_metrics.csv")
    save_config(vars(args), args.output, "training_config.json")

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train TorchLogix models")

    # Dataset and architecture
    parser.add_argument(
        "--dataset", type=str, choices=DATASET_CHOICES, required=True,
        help="Dataset to train on"
    )
    parser.add_argument(
        "--architecture", "-a", choices=ARCHITECTURE_CHOICES,
        default="randomly_connected", help="Model architecture"
    )

    # Model parameters
    parser.add_argument("--num_neurons", "-k", type=int, default=6000, help="Number of neurons")
    parser.add_argument("--num_layers", "-l", type=int, default=4, help="Number of layers")
    parser.add_argument("--tau", "-t", type=float, default=10, help="Softmax temperature")
    parser.add_argument("--grad-factor", type=float, default=1.0, help="Gradient factor")
    parser.add_argument(
        "--connections", type=str, default="random", choices=["random", "unique"],
        help="Connection type"
    )

    # Training parameters
    parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=2_000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--training-bit-count", "-c", type=int, default=32, help="Training bit count"
    )

    # Implementation
    parser.add_argument(
        "--implementation", type=str, default="cuda", choices=["cuda", "python"],
        help="Implementation to use (cuda is faster)"
    )

    # Data splitting
    parser.add_argument(
        "--valid-set-size", "-vss", type=float, default=0.2,
        help="Fraction of train set for validation"
    )

    # Output
    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/training/",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validation
    assert args.num_iterations % args.eval_freq == 0, (
        f"Number of iterations ({args.num_iterations}) must be divisible by "
        f"evaluation frequency ({args.eval_freq})"
    )

    run_training(args)


if __name__ == "__main__":
    main()