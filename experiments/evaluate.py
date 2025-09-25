#!/usr/bin/env python3
"""Evaluation script for trained TorchLogix models."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    DATASET_CHOICES, ARCHITECTURE_CHOICES, BITS_TO_TORCH_FLOATING_POINT_TYPE,
    IMPL_TO_DEVICE, setup_experiment, CreateFolder, save_metrics_csv, save_config,
    create_eval_functions, evaluate_model, get_model, load_dataset
)


def load_trained_model(model_path: Path, config_path: Path, device: str):
    """Load a trained model from checkpoint and config."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create args namespace from config
    class Args:
        pass

    args = Args()
    for key, value in config.items():
        setattr(args, key, value)

    # Get model architecture
    model, loss_fn, _ = get_model(args)
    model.to(device)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model, loss_fn, args


def comprehensive_evaluation(model, loader, eval_functions, device, mode="eval", num_classes=None):
    """Run comprehensive evaluation including detailed metrics."""
    orig_mode = model.training
    model.train(mode == "train")

    all_preds = []
    all_targets = []
    metrics = defaultdict(list)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)

            # Standard metrics
            for name, fn in eval_functions.items():
                metrics[name].append(fn(preds, y).to(torch.float32).mean().item())

            # Collect predictions for detailed analysis
            pred_classes = preds.argmax(-1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    model.train(orig_mode)

    # Aggregate metrics
    aggregated = {name: np.mean(vals) for name, vals in metrics.items()}

    # Detailed classification metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Classification report
    class_report = classification_report(
        all_targets, all_preds, output_dict=True, zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)

    return {
        'basic_metrics': aggregated,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist()
    }


def run_evaluation(args):
    """Run the evaluation."""
    # Setup experiment
    device = setup_experiment(args.seed, args.implementation)

    # Load trained model
    model, loss_fn, train_config = load_trained_model(
        args.model_path, args.config_path, device
    )

    # Load dataset (use same config as training)
    train_loader, validation_loader, test_loader = load_dataset(train_config)

    # Create evaluation functions
    eval_functions = create_eval_functions(loss_fn)

    results = {}

    print(f"Evaluating model: {args.model_path}")
    print(f"Dataset: {train_config.dataset}")
    print(f"Device: {device}")

    # Evaluate on different sets
    loaders = {"validation": validation_loader}

    if args.include_train:
        loaders["train"] = train_loader

    if args.include_test:
        loaders["test"] = test_loader

    # Evaluation modes
    eval_modes = ["eval"]  # Always evaluate in eval mode
    if args.train_mode:
        eval_modes.append("train")  # Also evaluate in train mode if requested

    for split_name, loader in loaders.items():
        if loader is None:
            continue

        print(f"\nEvaluating on {split_name} set...")
        results[split_name] = {}

        for mode in eval_modes:
            print(f"  Mode: {mode}")

            # Basic evaluation
            basic_metrics = evaluate_model(
                model, loader, eval_functions, mode=mode, device=device
            )

            print(f"  {split_name.capitalize()} ({mode}) - Loss: {basic_metrics['loss']:.4f}, "
                  f"Acc: {basic_metrics['acc']:.4f}")

            # Store results
            results[split_name][f"{mode}_metrics"] = basic_metrics

            # Comprehensive evaluation if requested
            if args.detailed:
                detailed_results = comprehensive_evaluation(
                    model, loader, eval_functions, device, mode=mode
                )
                results[split_name][f"{mode}_detailed"] = detailed_results

        # PackBits evaluation if requested (only in eval mode)
        if args.packbits_eval and args.implementation == "cuda":
            print(f"  Running PackBits evaluation on {split_name}...")
            packbits_metrics = evaluate_model(
                model, loader, eval_functions, mode="packbits", device=device
            )
            results[split_name]["packbits_metrics"] = packbits_metrics
            print(f"  {split_name.capitalize()} (PackBits) - Loss: {packbits_metrics['loss']:.4f}, "
                  f"Acc: {packbits_metrics['acc']:.4f}")

    # Save results
    save_config(results, args.output, "evaluation_results.json")

    # Save basic metrics CSV
    csv_metrics = {}
    for split_name, split_results in results.items():
        for result_key, metrics in split_results.items():
            if "metrics" in result_key and isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    csv_metrics[f"{split_name}_{result_key}_{metric_name}"] = value

    if csv_metrics:
        save_metrics_csv({0: csv_metrics}, args.output, "evaluation_summary.csv")

    # Save detailed classification reports if available
    if args.detailed:
        for split_name, split_results in results.items():
            for mode in eval_modes:
                detailed_key = f"{mode}_detailed"
                if detailed_key in split_results:
                    detailed_data = split_results[detailed_key]

                    # Save classification report
                    if "classification_report" in detailed_data:
                        class_report = detailed_data["classification_report"]
                        save_config(
                            class_report, args.output,
                            f"{split_name}_{mode}_classification_report.json"
                        )

                    # Save confusion matrix
                    if "confusion_matrix" in detailed_data:
                        conf_matrix = detailed_data["confusion_matrix"]
                        np.savetxt(
                            args.output / f"{split_name}_{mode}_confusion_matrix.csv",
                            conf_matrix, delimiter=",", fmt="%d"
                        )

    print(f"\nEvaluation completed! Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TorchLogix models")

    # Model and config paths
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--config-path", type=Path,
        help="Path to training config (.json file). If not provided, will look for config in model directory"
    )

    # Evaluation options
    parser.add_argument(
        "--include-train", action="store_true",
        help="Include training set in evaluation"
    )
    parser.add_argument(
        "--include-test", action="store_true",
        help="Include test set in evaluation"
    )
    parser.add_argument(
        "--train-mode", action="store_true",
        help="Also evaluate model in training mode (in addition to eval mode)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Run detailed evaluation with classification report and confusion matrix"
    )
    parser.add_argument(
        "--packbits-eval", action="store_true",
        help="Run PackBits evaluation (CUDA only)"
    )

    # Experiment setup
    parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--implementation", type=str, default="cuda", choices=["cuda", "python"],
        help="Implementation to use"
    )

    # Output
    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/evaluation/",
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    # Set default config path if not provided
    if args.config_path is None:
        args.config_path = args.model_path.parent / "training_config.json"

    # Validate paths
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    if not args.config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    run_evaluation(args)


if __name__ == "__main__":
    main()