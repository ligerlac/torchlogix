#!/usr/bin/env python3
"""Training script for TorchLogix models."""

import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Optional
from dataclasses import dataclass

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch
from tqdm import tqdm
import torchlogix
import torchlogix.models

from utils import (
    CreateFolder, save_metrics_csv, save_config, save_thresholds_csv,
    evaluate_model, get_model, load_dataset, load_n
)

def get_parser():
    parser = argparse.ArgumentParser(description="Train TorchLogix models")
    # Dataset and architecture
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar-10"],
        default="mnist", help="Dataset to train on"
    )
    parser.add_argument(
        "--architecture", "-a", choices=torchlogix.models.__dict__.keys(),
        default="DlgnMnistSmall", help="Model architecture. Must match dataset"
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
    parser.add_argument("--half-precision", action="store_true", 
                        help="Use half-precision (bfloat16) training to reduce memory usage and speed up training")
    parser.add_argument("--compile-model", action="store_true", 
                        help="Use TorchScript to compile the model for faster inference (only works for conv models)")

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
        "--connections-temperature", type=float, default=0.001,
        help="Temperature for softmax in learnable connections"
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
        "--parametrization", type=str, default="raw", choices=["raw", "warp", "light"],
        help="Parametrization to use"
    )
    parser.add_argument(
        "--parametrization-temperature", type=float, default=0.1,
        help="Temperature for sigmoid/softmax in parametrization"
    )
    parser.add_argument(
        "--forward-sampling", type=str, default="soft", choices=["soft", "hard", "gumbel_soft", "gumbel_hard"],
        help="Sampling method in forward pass during training"
    )
    parser.add_argument(
        "--weight-init", type=str, default="residual", choices=["residual", "random", "residual-catalog"],
        help="Initialization method for model weights"
    )
    parser.add_argument(
        "--residual-probability", type=float, default=0.951,
        help="Parameter for residual weight initialization. " \
        "Corresponds to probability of a LUT entry corresponding to identity LUT entry."
    )

    # Binarization parameters
    parser.add_argument(
        "--binarization-num-batches", type=int, default=100,
        help="Number of batches for initializing thresholds in binarization"
    )
    parser.add_argument(
        "--binarization", type=str, default="fixed", choices=["dummy", "fixed", "soft", "learnable"],
        help="Binarization method for input data"
    )
    parser.add_argument(
        "--binarization-init", type=str, default="uniform", choices=["uniform", "distributive"],
        help="Method to find initial thresholds for binarization"
    )
    parser.add_argument(
        "--binarization-per", type=str, default="global", choices=["global", "feature", "channel"],
        help="Binarization thresholds global, per channel, or per feature"
    )
    parser.add_argument(
        "--binarization-temperature", type=float, default=0.01,
        help="Temperature for sampling in learnable binarization"
    )
    parser.add_argument(
        "--binarization-temperature-softplus", type=float, default=0.01,
        help="Temperature for softplus in learnable binarization"
    )
    parser.add_argument(
        "--binarization-learning-rate", type=float, default=None,
        help="Learning rate for binarization (as fraction of main learning rate). If None, uses main learning rate."
    )
    parser.add_argument(
        "--binarization-forward-sampling", type=str, default="soft", choices=["soft", "hard", "gumbel_soft", "gumbel_hard"],
        help="Sampling method in forward pass during training for learnable binarization"
    )

    return parser


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
            self.scheduler.step(ctx.metrics["val_loss_discrete"])
        elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step()

    @classmethod
    def from_args(cls, optimizer, args):
        if args.lr_schedule is None:
            scheduler = None
        elif args.lr_schedule == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.lr_patience#, verbose=False
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
    val_acc = ctx.metrics.get("val_acc_discrete", 0.0)
    if not hasattr(save_best_model, "best_val_acc"):
        save_best_model.best_val_acc = 0.0

    if val_acc > save_best_model.best_val_acc:
        save_best_model.best_val_acc = val_acc
        model_path = f"{output_dir}/best_model.pt"
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
    thresholds = torchlogix.layers.Binarization.get_initial_thresholds(
        data_set,
        num_bits=model_cls.n_input_bits,
        one_per=args.binarization_per,
        method=args.binarization_init
    )

    print("Initial thresholds:", thresholds)

    # Get model, loss, and optimizer
    model= get_model(thresholds, args)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    model.to(args.device)
    print(model)

    print(f"model.parameters(): {[p.shape for p in model.parameters()]}")

    if args.compile_model:
        print("compiling...")
        model.compile(dynamic=True)  # speedup w/ JIT compilation

    # Loss function for classification tasks like MNIST and CIFAR-10
    loss_fn = torch.nn.CrossEntropyLoss()
    # Create evaluation functions
    eval_functions = {
        "loss": loss_fn,
        "acc": lambda preds, y: (preds.argmax(-1) == y).to(torch.float32).mean(),
    }

    # Set up optimizer with optional separate learning rate for binarization parameters
    params_list = []
    binarization_params = []
    if args.binarization_learning_rate and isinstance(model[0], torchlogix.layers.LearnableBinarization):
        binarization_params += list(model[0].parameters())
        params_list += [{'params': binarization_params, 'lr': args.binarization_learning_rate * args.learning_rate}]
    else:
        if args.binarization_learning_rate:
            print("Warning: binarization_learning_rate specified but the model does not use LearnableBinarization. Ignoring this parameter.")
    other_params = [p for p in model.parameters() if p not in set(binarization_params)]
    params_list += [{'params': other_params, 'lr': args.learning_rate}]
    optimizer = torch.optim.Adam(params_list)

    # Training tracking
    metrics = defaultdict(dict)
    best_val_acc = 0.0

    learning_rate_scheduler = LearningRateSchedulerCallback.from_args(optimizer, args)
    callbacks.append(learning_rate_scheduler)

    print(f"Starting training for {args.num_iterations} iterations...")
    print(f"Model: {args.architecture}, Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    
    if args.output is not None:
        save_config(vars(args), args.output, "training_config.json")

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="Training",
        total=args.num_iterations,
        mininterval=1,
    )
    running_train_loss, n = 0.0, 0
    for i, (x, y) in pbar:
        x = x.to(args.device)
        y = y.to(args.device)

        dtype = torch.bfloat16 if args.half_precision else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype):
            model.train()
            x = model(x)
            loss = loss_fn(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        n += y.size(0)
        running_train_loss += loss

        if i % 100 == 0:
            pbar.set_postfix(loss=f"{loss:.4f}")

        # Evaluation
        if (args.eval_freq > 0 and ((i + 1) % args.eval_freq == 0)):
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
                {"train_loss": running_train_loss.cpu().detach().item() / n * len(validation_loader)}
        
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
        lambda ctx: save_metrics_csv(ctx.step, ctx.metrics, args.output),
        lambda ctx: save_thresholds_csv(ctx.step, thresholds=ctx.model[0].get_thresholds().detach(), 
                                        output_path=args.output) if hasattr(ctx.model[0], "get_thresholds") else None
    ]

    # Pretty print args
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    run_training(args, call_backs)


if __name__ == "__main__":
    main()
