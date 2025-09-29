#!/usr/bin/env python3
"""Training script for TorchLogix models."""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
from tqdm import tqdm

from utils import (
    DATASET_CHOICES, ARCHITECTURE_CHOICES, BITS_TO_TORCH_FLOATING_POINT_TYPE,
    IMPL_TO_DEVICE, setup_experiment, CreateFolder, save_metrics_csv, save_config,
    create_eval_functions, evaluate_model, train, get_model, load_dataset, load_n
)

import torchlogix


def run_training(args):
    """Run the training loop."""
    # Setup experiment
    device = setup_experiment(args.seed, args.implementation)

    # Load data (omit test set during training)
    train_loader, validation_loader, _ = load_dataset(args)

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

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="Training",
        total=args.num_iterations,
    )
    for i, (x, y) in pbar:
        # Move data to device
        # x = x.bool().float()
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)
        y = y.to(device)

        # import numpy as np
        # print(np.unique(x.cpu().numpy()))

        # Training step
        loss = train(model, x, y, loss_fn, optim)
        pbar.set_postfix(loss=f"{loss:.4f}")

        if args.temp_decay is not None:
            temperature = np.exp(- i / args.num_iterations * args.temp_decay)
            for layer in model:
                if isinstance(layer, torchlogix.layers.LogicConv2d) or isinstance(layer, torchlogix.layers.LogicDense):
                    layer.temperature = temperature
            pbar.set_postfix(loss=f"{loss:.4f}", temp=f"{temperature:.4f}")

        # Log training loss
        metrics[i + 1] = {"train_loss": loss}

        # Evaluation
        if (i + 1) % args.eval_freq == 0:
            print(f"\nEvaluation at iteration {i + 1}")

            # with torch.no_grad():
            #     for i, layer in enumerate(model):
            #         if isinstance(layer, torchlogix.layers.LogicConv2d):
            #             layer_type = "Conv"
            #             all_params = []
            #             for param_list in layer.tree_weights:
            #                 for param in param_list:
            #                     all_params.append(param.data.detach().cpu().numpy())
            #             all_params = np.concatenate([p for p in all_params])
            #         elif isinstance(layer, torchlogix.layers.LogicDense):
            #             layer_type = "Dense"
            #             all_params = layer.weight.data.detach().cpu().numpy()
            #         else:
            #             continue
            #         m, s = all_params.mean(axis=0), all_params.std(axis=0)
            #         print(f"{layer_type} Layer {m[0]:.3f} +- {s[0]:.3f} | {m[1]:.3f} +- {s[1]:.3f} | {m[2]:.3f} +- {s[2]:.3f} | {m[3]:.3f} +- {s[3]:.3f}")

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
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--temp-decay", "-td", type=float, default=None,
                         help="Temperature decay, e.g. 4 (only applicable to walsh-parametrized models)")
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=2_000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--training-bit-count", "-c", type=int, default=32, help="Training bit count"
    )

    parser.add_argument(
        "--implementation", type=str, default="cuda", choices=["cuda", "python"],
        help="Implementation to use (cuda is faster)"
    )
    parser.add_argument(
        "--parametrization", type=str, default="raw", choices=["raw", "walsh"],
        help="Parametrization to use"
    )

    parser.add_argument(
        "--valid-set-size", "-vss", type=float, default=0.2,
        help="Fraction of train set for validation"
    )

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/training/",
        help="Output directory for results"
    )

    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for sampling in forward pass for walsh parametrization"
    )

    parser.add_argument(
        "--forward-sampling", type=str, default="soft", choices=["soft", "hard", "gumbel_soft", "gumbel_hard"],
        help="Sampling method in forward pass during training"
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