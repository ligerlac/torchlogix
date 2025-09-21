#!/usr/bin/env python3
"""Model compilation script for TorchLogix models."""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from shared_config import setup_experiment
from utils import CreateFolder, save_metrics_csv, save_config
from model_selection import get_model
from loading import load_dataset
from torchlogix import CompiledLogicNet


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


def benchmark_model(model, test_loader, device, num_runs=3):
    """Benchmark model inference speed."""
    model.eval()
    times = []

    # Warmup
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 2:  # Just a few warmup runs
                break
            x = x.to(device)
            _ = model(x)

    # Actual timing
    with torch.no_grad():
        for run in range(num_runs):
            start_time = time.time()
            total_samples = 0

            for x, y in test_loader:
                x = x.to(device)
                _ = model(x)
                total_samples += x.shape[0]

            elapsed = time.time() - start_time
            times.append(elapsed)

    avg_time = np.mean(times)
    samples_per_second = total_samples / avg_time

    return {
        "avg_time": avg_time,
        "samples_per_second": samples_per_second,
        "total_samples": total_samples
    }


def benchmark_compiled_model(compiled_model, test_data, num_runs=3):
    """Benchmark compiled model inference speed."""
    times = []

    # Warmup
    for _ in range(2):
        _ = compiled_model(test_data[:100] if len(test_data) > 100 else test_data)

    # Actual timing
    for run in range(num_runs):
        start_time = time.time()
        _ = compiled_model(test_data)
        elapsed = time.time() - start_time
        times.append(elapsed)

    avg_time = np.mean(times)
    samples_per_second = len(test_data) / avg_time

    return {
        "avg_time": avg_time,
        "samples_per_second": samples_per_second,
        "total_samples": len(test_data)
    }


def run_compilation(args):
    """Run model compilation and benchmarking."""
    # Setup experiment
    device = setup_experiment(args.seed, "python")  # Compilation uses CPU

    # Load trained model
    model, loss_fn, train_config = load_trained_model(
        args.model_path, args.config_path, "cpu"  # Load on CPU for compilation
    )

    print(f"Compiling model: {args.model_path}")
    print(f"Dataset: {train_config.dataset}")

    # Load test dataset for evaluation
    _, _, test_loader = load_dataset(train_config)

    if test_loader is None:
        raise ValueError("Test loader is required for compilation benchmarking")

    # Prepare test data for compiled model
    test_data_list = []
    test_labels_list = []

    with torch.no_grad():
        for x, y in test_loader:
            # Flatten and convert to boolean
            x_flat = torch.nn.Flatten()(x).bool().numpy()
            test_data_list.append(x_flat)
            test_labels_list.append(y.numpy())

    test_data = np.vstack(test_data_list)
    test_labels = np.concatenate(test_labels_list)

    print(f"Test data shape: {test_data.shape}")

    # Benchmark original model if requested
    original_benchmark = None
    if args.benchmark_original:
        print("\nBenchmarking original PyTorch model...")
        original_benchmark = benchmark_model(model, test_loader, "cpu", args.num_runs)
        print(f"Original model - Avg time: {original_benchmark['avg_time']:.4f}s, "
              f"Samples/sec: {original_benchmark['samples_per_second']:.0f}")

    compilation_results = {}

    # Compile with different settings
    for opt_level in args.opt_levels:
        for num_bits in args.bit_counts:
            print(f"\n" + "="*50)
            print(f"Compiling with opt_level={opt_level}, num_bits={num_bits}")
            print("="*50)

            try:
                # Create output directory for this configuration
                config_output = args.output / f"opt{opt_level}_bits{num_bits}"
                config_output.mkdir(exist_ok=True)

                save_lib_path = config_output / f"model_opt{opt_level}_{num_bits}bits.so"

                # Compile model
                start_compile = time.time()

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler=args.compiler,
                    verbose=args.verbose,
                )

                compiled_model.compile(
                    opt_level=opt_level,
                    save_lib_path=str(save_lib_path),
                    verbose=args.verbose,
                )

                compile_time = time.time() - start_compile

                print(f"Compilation completed in {compile_time:.2f}s")

                # Test accuracy
                print("Testing compiled model accuracy...")
                output = compiled_model(test_data, verbose=args.verbose)
                predictions = output.argmax(-1)
                accuracy = (predictions == test_labels).mean()

                print(f"Compiled model accuracy: {accuracy:.4f}")

                # Benchmark compiled model
                print("Benchmarking compiled model...")
                compiled_benchmark = benchmark_compiled_model(
                    compiled_model, test_data, args.num_runs
                )

                print(f"Compiled model - Avg time: {compiled_benchmark['avg_time']:.4f}s, "
                      f"Samples/sec: {compiled_benchmark['samples_per_second']:.0f}")

                # Calculate speedup
                speedup = None
                if original_benchmark:
                    speedup = (original_benchmark["samples_per_second"] /
                             compiled_benchmark["samples_per_second"])

                # Store results
                config_key = f"opt{opt_level}_bits{num_bits}"
                compilation_results[config_key] = {
                    "opt_level": opt_level,
                    "num_bits": num_bits,
                    "compile_time": compile_time,
                    "accuracy": float(accuracy),
                    "benchmark": compiled_benchmark,
                    "speedup_vs_original": speedup,
                    "library_path": str(save_lib_path)
                }

                # Save individual config results
                save_config(
                    compilation_results[config_key],
                    config_output,
                    "compilation_results.json"
                )

            except Exception as e:
                print(f"Compilation failed for opt_level={opt_level}, num_bits={num_bits}")
                print(f"Error: {e}")

                config_key = f"opt{opt_level}_bits{num_bits}"
                compilation_results[config_key] = {
                    "opt_level": opt_level,
                    "num_bits": num_bits,
                    "error": str(e),
                    "success": False
                }

    # Save overall results
    final_results = {
        "model_path": str(args.model_path),
        "original_benchmark": original_benchmark,
        "compilation_results": compilation_results,
        "compilation_settings": {
            "compiler": args.compiler,
            "opt_levels": args.opt_levels,
            "bit_counts": args.bit_counts,
        }
    }

    save_config(final_results, args.output, "compilation_summary.json")

    # Create CSV summary
    csv_data = {}
    for config_key, results in compilation_results.items():
        if "error" not in results:
            csv_data[config_key] = {
                "opt_level": results["opt_level"],
                "num_bits": results["num_bits"],
                "compile_time": results["compile_time"],
                "accuracy": results["accuracy"],
                "samples_per_second": results["benchmark"]["samples_per_second"],
                "speedup": results.get("speedup_vs_original", None)
            }

    if csv_data:
        save_metrics_csv(csv_data, args.output, "compilation_summary.csv")

    print(f"\n" + "="*50)
    print("Compilation completed!")
    print(f"Results saved to: {args.output}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Compile trained TorchLogix models")

    # Model and config paths
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--config-path", type=Path,
        help="Path to training config (.json file). If not provided, will look for config in model directory"
    )

    # Compilation settings
    parser.add_argument(
        "--compiler", type=str, default="gcc", choices=["gcc", "clang"],
        help="C compiler to use"
    )
    parser.add_argument(
        "--opt-levels", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Optimization levels to test"
    )
    parser.add_argument(
        "--bit-counts", type=int, nargs="+", default=[32, 64],
        help="Bit counts to test"
    )

    # Benchmarking
    parser.add_argument(
        "--benchmark-original", action="store_true",
        help="Also benchmark the original PyTorch model"
    )
    parser.add_argument(
        "--num-runs", type=int, default=3,
        help="Number of benchmark runs for averaging"
    )

    # Experiment setup
    parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose compilation output"
    )

    # Output
    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/compilation/",
        help="Output directory for compilation results"
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

    run_compilation(args)


if __name__ == "__main__":
    main()