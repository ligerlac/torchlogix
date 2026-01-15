#!/usr/bin/env python3
"""Training script for TorchLogix models."""

import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Optional
from dataclasses import dataclass
import math
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import torch
from tqdm import tqdm

from torchlogix.layers import Binarization
from utils import (
    DATASET_CHOICES, ARCHITECTURE_CHOICES,
    IMPL_TO_DEVICE, setup_experiment, CreateFolder, save_metrics_csv, save_config,
    create_eval_functions, evaluate_model, train, get_model, load_dataset, load_n, print_memory_usage
)

import torchlogix


def get_parser():
    parser = argparse.ArgumentParser(description="Train TorchLogix models")
    # Dataset and architecture
    parser.add_argument(
        "--dataset", type=str, choices=DATASET_CHOICES,
        default="mnist", help="Dataset to train on"
    )
    parser.add_argument(
        "--architecture", "-a", choices=torchlogix.models.__dict__.keys(),
        default="DlgnMnistSmall", help="Model architecture. Must match dataset"
    )
    parser.add_argument(
        "--implementation", type=str, default="python", choices=["cuda", "python"],
        help="Implementation to use (cuda is faster)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )

    # Training parameters
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=2_000, 
        help="Evaluation frequency. Evaluation is deactivated if set to 0."
    )
    parser.add_argument(
        "--valid-set-size", "-vss", type=float, default=0.2,
        help="Fraction of train set for validation"
    )

    # Learning rate parameters
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--lr-schedule", type=str, choices=[None, "CosineAnnealingWarmRestarts", "ReduceLROnPlateau"],
        default=None, help="Learning rate scheduling strategy"
    )
    parser.add_argument(
        "--lr-reduction-factor", "-lrf", type=float, default=0.2,
        help="If lr-schedule is ReduceLROnPlateau, factor by which LR will be reduced. "
    )
    parser.add_argument(
        "--lr-patience", "-lrp", type=int, default=10, 
        help="If lr-schedule is ReduceLROnPlateau, patience for LR reduction." \
             "If CosineAnnealingWarmRestarts, length of each cycle." \
             "Counted in number of evaluations."
    )

    parser.add_argument("--temp-decay", "-td", type=float, default=None,
                         help="Temperature decay, e.g. 4 (only applicable to walsh-parametrized models)")
    parser.add_argument("--half-precision", action="store_true", 
                        help="Use half-precision (bfloat16) training to reduce memory usage and speed up training")

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/training/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", type=int, default=0, choices=[0, 1],
        help="Verbosity during training, allowed only for lut_rank=2. 0 = silent, 1 = verbose"
    )

    # Connection parameters
    parser.add_argument(
        "--connections-init-method", type=str, choices=["random", "random-unique"],
        default="random", help="Connection initilization strategy"
    )
    parser.add_argument(
        "--connections", type=str, choices=["fixed", "learnable"],
        default="fixed", help="Connection strategy"
    )
    parser.add_argument(
        "--connections-num-candidates", type=int,
        default=-1, help="Number of candidates for learnable connections"
    )
    parser.add_argument(
        "--connections-temperature", type=float, default=0.1,
        help="Temperature for softmax in learnable connections"
    )
    parser.add_argument(
        "--connections-temperature-decay", type=str, default=None, choices=[None, "linear", "exponential"],
        help="Temperature decay for softmax in learnable connections"
    )
    parser.add_argument(
        "--connections-temperature-end", type=float, default=0.01,
        help="Temperature end value of decay for softmax in learnable connections"
    )
    parser.add_argument(
        "--connections-gumbel", action="store_false", 
        help="Flag for using Gumbel sampling for softmax. "
    )

    # Parametrization parameters
    parser.add_argument(
        "--lut-rank", type=int, default=2, choices=[2, 4, 6],
        help="Number of inputs to each LUT node"
    )
    parser.add_argument(
        "--parametrization", type=str, default="raw", choices=["raw", "walsh", "light"],
        help="Parametrization to use"
    )
    parser.add_argument(
        "--parametrization-temperature", type=float, default=0.1,
        help="Temperature for sigmoid in walsh parametrization"
    )
    parser.add_argument(
        "--parametrization-temperature-decay", type=str, default=None, choices=[None, "linear", "exponential"],
        help="Temperature decay for sigmoid in walsh parametrization"
    )
    parser.add_argument(
        "--parametrization-temperature-end", type=float, default=0.01,
        help="Final temperature of decay for sigmoid in walsh parametrization"
    )
    parser.add_argument(
        "--forward-sampling", type=str, default="soft", choices=["soft", "hard", "gumbel_soft", "gumbel_hard"],
        help="Sampling method in forward pass during training"
    )
    parser.add_argument(
        "--weight-init", type=str, default="residual", choices=["residual", "random"],
        help="Initialization method for model weights"
    )
    parser.add_argument(
        "--residual-param", type=float, default=5.0,
        help="Parameter for residual weight initialization. " \
        "Corresponds to percentage of LUTs initialized to identity gate function."
    )

    # Regularization parameters
    parser.add_argument(
        "--regularization-weight", type=float, default=0.0,
        help="Regularization strength"
    )
    parser.add_argument(
        "--regularization-weight-increase", type=str, default=None, choices=[None, "linear", "exponential"],
        help="Regularization weight increase method"
    )
    parser.add_argument(
        "--regularization-weight-end", type=float, default=0.0,
        help="Final regularization strength"
    )
    parser.add_argument(
        "--regularization", type=str, default=None, choices=[None, "abs_sum", "L2"],
        help="Regularization method"
    )
    parser.add_argument(
        "--weight-rescale", type=str, default=None, choices=[None, "clip", "abs_sum", "L2"],
        help="Weight rescaling for each layer after each training step"
    )

    # Binarization parameters
    parser.add_argument(
        "--binarization-num-batches", type=int, default=100,
        help="Number of batches for initializing thresholds in binarization"
    )
    parser.add_argument(
        "--binarization", type=str, default="uniform", choices=["dummy", "uniform", "learnable"],
        help="Binarization method for input data"
    )
    parser.add_argument(
        "--binarization-feature-wise", action="store_true", 
        help="Flag for feature-wise binarization thresholds"
    )
    parser.add_argument(
        "--binarization-temperature", type=float, default=0.1,
        help="Temperature for sampling in learnable binarization"
    )
    parser.add_argument(
        "--binarization-temperature-decay", type=str, default=None, choices=[None, "linear", "exponential"],
        help="Temperature decay for sigmoid in walsh parametrization"
    )
    parser.add_argument(
        "--binarization-temperature-end", type=float, default=0.01,
        help="Final temperature of decay for sigmoid in walsh parametrization"
    )
    parser.add_argument(
        "--binarization-temperature-softplus", type=float, default=1.0,
        help="Temperature for softplus in learnable binarization"
    )

    return parser


class DecaySchedule:
    def __init__(self, initial_value: float, final_value: float, decay_type: str, total_steps: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_type = decay_type
        self.total_steps = total_steps
        assert decay_type in [None, "linear", "exponential"], f"Unknown decay type: {decay_type}"

    def __call__(self, step) -> float:
        if self.decay_type is None:
            value = self.initial_value
        elif self.decay_type == "linear":
            value = self.initial_value - (self.initial_value - self.final_value) * (step / self.total_steps)
            value = max(value, self.final_value)
        elif self.decay_type == "exponential":
            decay_rate = math.log(self.final_value / self.initial_value) / self.total_steps
            value = self.initial_value * math.exp(decay_rate * step)
            value = max(value, self.final_value)
        return value


@dataclass
class CallbackContext:
    """Context passed to training callbacks."""
    step: int
    metrics: dict  # Required: val_loss, train_loss, etc.
    model: Optional[torch.nn.Module] = None  # Optional for advanced use


class LearningRateSchedulerCallback:
    """Wrapper for learning rate schedulers to be used as callbacks."""
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, ctx: CallbackContext):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(ctx.metrics["val_loss"])
        elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step()

    @classmethod
    def from_args(cls, optimizer, args):
        if args.lr_schedule is None:
            scheduler = None
        elif args.lr_schedule == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.lr_patience, verbose=False
            )
        elif args.lr_schedule == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=args.lr_patience, T_mult=1, eta_min=0.0, last_epoch=-1
            )
        else:
            raise ValueError(f"Unknown learning rate schedule: {args.lr_schedule}")
        return cls(scheduler)


def save_best_model(ctx: CallbackContext, output_dir: Path):
    """Callback to save the best model based on validation accuracy."""
    val_acc = ctx.metrics.get("val_accuracy_eval", 0.0)
    if not hasattr(save_best_model, "best_val_acc"):
        save_best_model.best_val_acc = 0.0

    if val_acc > save_best_model.best_val_acc:
        save_best_model.best_val_acc = val_acc
        model_path = output_dir / "best_model.pt"
        torch.save(ctx.model.state_dict(), model_path)
        print(f"New best model saved with val_accuracy: {val_acc:.4f} at step {ctx.step}")


def run_training(args, callbacks=None):
    """Run the training loop."""
    if callbacks is None:
        callbacks = []
    # Setup experiment
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    torch.set_num_threads(1)

    # Load data (omit test set during training)
    train_loader, validation_loader, _ = load_dataset(args)

    # Initial thresholds
    data_set = torch.cat(tuple([batch[0] for batch in load_n(train_loader, args.binarization_num_batches)]))
    model_cls = torchlogix.models.__dict__[args.architecture]
    thresholds = Binarization.get_initial_thresholds(
        data_set,
        num_bits=model_cls.n_input_bits,
        feature_wise=args.binarization_feature_wise,
        method=args.binarization
    )

    # Get model, loss, and optimizer
    model, loss_fn, optim = get_model(thresholds, args)
    model.to(args.device)
    # model.compile()  # factor ~2 speedup w/ JIT compilation

    # Create evaluation functions
    eval_functions = create_eval_functions(loss_fn)

    # Training tracking
    metrics = defaultdict(dict)
    best_val_acc = 0.0

    # Decay schedules
    total_steps = args.num_iterations
    parametrization_schedule = DecaySchedule(
        initial_value=args.parametrization_temperature,
        final_value=args.parametrization_temperature_end,
        decay_type=args.parametrization_temperature_decay,
        total_steps=total_steps
    )
    connection_schedule = DecaySchedule(
        initial_value=args.connections_temperature,
        final_value=args.connections_temperature_end,
        decay_type=args.connections_temperature_decay,
        total_steps=total_steps
    )
    regularization_schedule = DecaySchedule(
        initial_value=args.regularization_weight,
        final_value=args.regularization_weight_end,
        decay_type=args.regularization_weight_increase,
        total_steps=total_steps
    )
    binarization_schedule = DecaySchedule(
        initial_value=args.binarization_temperature,
        final_value=args.binarization_temperature_end,
        decay_type=args.binarization_temperature_decay,
        total_steps=total_steps
    )

    learning_rate_scheduler = LearningRateSchedulerCallback.from_args(optim, args)
    callbacks.append(learning_rate_scheduler)

    print(f"Starting training for {args.num_iterations} iterations...")
    print(f"Model: {args.architecture}, Dataset: {args.dataset}")
    print(f"Device: {args.device}, Implementation: {args.implementation}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    
    if args.output is not None:
        save_config(vars(args), args.output, "training_config.json")

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="Training",
        total=args.num_iterations,
    )
    running_train_loss, n = 0.0, 0
    for i, (x, y) in pbar:
        x = x.to(args.device)
        y = y.to(args.device)

        dtype = torch.bfloat16 if args.half_precision else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = train(
                model, x, y, loss_fn, optim,
                regularization_method=args.regularization,
                regularization_weight=regularization_schedule(i),
            )

        print_memory_usage("After training step")

        n += y.size(0)
        running_train_loss += loss

        # Update layer parameters according to schedules
        model[0].temperature_sampling = binarization_schedule(i)
        for idx in range(1, len(model)):
            layer = model[idx]
            if hasattr(layer, "rescale_weights") and args.weight_rescale is not None:
                layer.rescale_weights(args.weight_rescale)
            if hasattr(layer, "parametrization") and args.parametrization_temperature_decay is not None:
                # Parametrization temperature decay
                layer.parametrization.temperature = parametrization_schedule(i)
            if hasattr(layer, "connections") and args.connections == "learnable" and args.connections_temperature_decay is not None:
                # Connection temperature decay
                layer.connections.temperature = connection_schedule(i)

        if i % 10 == 0:
            pbar.set_postfix(loss=f"{loss:.4f}")

        # Evaluation
        if (args.eval_freq > 0 and ((i + 1) % args.eval_freq == 0)):
            # if args.reg_lambda > 0.0:
            if args.verbose == 1:
                print(f"\nEvaluation at iteration {i + 1}")          

            # Evaluate on validation set
            discrete_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="eval", device=args.device
            )
            relaxed_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="train", device=args.device
            )

            metrics = \
                {f"val_{k}_discrete": v for k, v in discrete_metrics.items()} | \
                {f"val_{k}_relaxed": v for k, v in relaxed_metrics.items()} | \
                {"train_loss": running_train_loss / n * len(validation_loader)}

            print(f"Iteration {i + 1:6d} | " +
                  " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

            running_train_loss, n = 0.0, 0

            ctx = CallbackContext(
                step=i + 1,
                metrics=metrics,
                model=model
            )

            for cb in callbacks:
                cb(ctx)

    # Save final model
    if args.output is not None:
        torch.save(model.state_dict(), f"{args.output}/final_model.pt")

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {args.output}")

    return metrics


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Validation
    if args.eval_freq > 0:
        assert args.num_iterations % args.eval_freq == 0, (
            f"Number of iterations ({args.num_iterations}) must be divisible by "
            f"evaluation frequency ({args.eval_freq})"
        )

    call_backs = [
        lambda ctx: save_best_model(ctx, args.output),
        lambda ctx: save_metrics_csv(ctx.metrics, args.output),
    ]

    run_training(args, call_backs)


if __name__ == "__main__":
    main()
