#!/usr/bin/env python3
"""Training script for TorchLogix models."""

import argparse
import random
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


def get_gate_ids_walsh(model):
    device = next(model.parameters()).device
    binary_inputs = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32, device=device)
    truth_tables = (
        (0,0,0,0), (1,1,1,1), (0,0,0,1), (0,1,1,1), (0,1,1,0), (1,0,0,1), (1,1,1,0), (1,0,0,0), \
        (0,0,1,0), (0,1,0,0), (0,0,1,1), (1,1,0,0), (0,1,0,1), (1,0,1,0), (1,1,0,1), (1,0,1,1)
    )
    truth_tables = torch.tensor(truth_tables, dtype=torch.float32, device=device)  # Shape: (16, 4)
    gate_ids = []
    for param in model.parameters():
        inputs_expanded = binary_inputs.unsqueeze(0)  # Shape: (1, 4, 2)

        # Now the computation will broadcast correctly
        linear_preds = (param[:, 0].unsqueeze(1) + 
                        param[:, 1].unsqueeze(1) * inputs_expanded[:, :, 0] + 
                        param[:, 2].unsqueeze(1) * inputs_expanded[:, :, 1] + 
                        param[:, 3].unsqueeze(1) * inputs_expanded[:, :, 0] * inputs_expanded[:, :, 1])

        preds = (linear_preds > 0.0).float()

        # Compare preds with all truth tables to find ids
        dists = ((preds.unsqueeze(1) - truth_tables.unsqueeze(0)).abs().sum(dim=-1)).cpu().numpy()
        ids = dists.argmin(axis=1)

        # Compute gate outputs based on counts
        gate_ids.append(ids)

    return np.concatenate(gate_ids)


def get_gate_ids_raw(model):
    gate_ids = []
    for param in model.parameters():
        # argmax over all 16 weights
        ids = param.argmax(dim=-1).cpu().numpy()
        gate_ids.append(ids)

    return np.concatenate(gate_ids)


def analyze_nodes_closest_to_half(model, input_data, top_n=5):
    """
    Find nodes with activations closest to 0.5 and print their parameters
    """
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):  # Layers with parameters
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_data)
    
    # Collect all nodes closest to 0.5 across all layers
    all_candidates = []
    
    # for layer_name, activation in activations.items():
    for layer_name, activation in activations.items():
        module = dict(model.named_modules())[layer_name]
        activation_np = activation.cpu().numpy()

        batch_size, num_nodes = activation_np.shape

        for node_idx in range(num_nodes):
            node_activations = activation_np[:, node_idx]                        
            target_vector = np.full_like(node_activations, 0.5)
            l2_distance = np.linalg.norm(node_activations - target_vector)
            mean_activation = node_activations.mean()
            all_candidates.append({
                'activations': node_activations,
                'layer_name': layer_name,
                'node_idx': node_idx,
                'mean_activation': mean_activation,
                'l2_distance': l2_distance,
                'module': module
            })
        # look only at first layer
        break

    # Sort by distance from 0.5
    all_candidates.sort(key=lambda x: x['l2_distance'])
    
    # Print top N closest nodes
    print(f"\nTop {top_n} nodes closest to 0.5 across all layers:")
    print("=" * 60)
    
    for i, c in enumerate(all_candidates[:top_n]):
        print(f"\n{i+1}. Layer {c['layer_name']}, Node {c['node_idx']}")
        print(f"   Mean Activation: {c['mean_activation']:.6f} (L2 distance from 0.5: {c['l2_distance']:.6f})")
        print(f"   Weights: {c['module'].weight[c['node_idx']].detach().cpu().numpy()}")

    best_candidate = all_candidates[0]
    activations = best_candidate['activations']
    h = hist.Hist.new.Regular(10, 0., 1., name="x").Double()
    h.fill(activations)
    print(h)

    # Print top N furthest nodes
    print(f"\nTop {top_n} nodes furthest from 0.5 across all layers:")
    print("=" * 60)

    for i, c in enumerate(all_candidates[-top_n:]):
        print(f"\n{i+1}. Layer {c['layer_name']}, Node {c['node_idx']}")
        print(f"   Mean Activation: {c['mean_activation']:.6f} (L2 distance from 0.5: {c['l2_distance']:.6f})")
        print(f"   Weights: {c['module'].weight[c['node_idx']].detach().cpu().numpy()}")

    best_candidate = all_candidates[-1]
    activations = best_candidate['activations']
    h = hist.Hist.new.Regular(10, 0., 1., name="x").Double()
    h.fill(activations)
    print(h)
    
    # Clean up
    for hook in hooks:
        hook.remove()


def run_training(args):
    """Run the training loop."""
    # Setup experiment
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    # Load data (omit test set during training)
    train_loader, validation_loader, _ = load_dataset(args)

    # Get model, loss, and optimizer
    model, loss_fn, optim = get_model(args)
    model.to(args.device)

    # Create evaluation functions
    eval_functions = create_eval_functions(loss_fn)

    # Training tracking
    metrics = defaultdict(list)
    best_val_acc = 0.0

    print(f"Starting training for {args.num_iterations} iterations...")
    print(f"Model: {args.architecture}, Dataset: {args.dataset}")
    print(f"Device: {args.device}, Implementation: {args.implementation}")
    save_config(vars(args), args.output, "training_config.json")

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="Training",
        total=args.num_iterations,
    )
    for i, (x, y) in pbar:
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(args.device)
        y = y.to(args.device)

        # Training step
        loss = train(model, x, y, loss_fn, optim)#, reg_lambda=args.reg_lambda)
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
        if ((i + 1) % args.eval_freq == 0) or (i == 0):
            # if args.reg_lambda > 0.0:
            print(f"\nEvaluation at iteration {i + 1}")

            import hist
            all_weights = []
            for param in model.parameters():
                all_weights.append(param.view(-1).detach().cpu().numpy())
            all_weights = np.concatenate(all_weights)
            h = hist.Hist.new.Regular(30, -3, 3, name="x").Double()
            h.fill(all_weights)
            print(h)

            if args.parametrization == "walsh":
                gate_ids = get_gate_ids_walsh(model)
            elif args.parametrization == "raw":
                gate_ids = get_gate_ids_raw(model)
            
            h = hist.Hist.new.Regular(16, 0, 16, name="x").Double()
            h.fill(gate_ids)
            print(h)

            print("Gate counts:", gate_ids)            
            # analyze_nodes_closest_to_half(model, x, top_n=10)

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
                model, validation_loader, eval_functions, mode="eval", device=args.device
            )
            train_metrics = evaluate_model(
                model, validation_loader, eval_functions, mode="train", device=args.device
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

            if args.temp_decay is not None:
                temperature = np.exp(- i / args.num_iterations * args.temp_decay)
                for layer in model:
                    if isinstance(layer, torchlogix.layers.LogicConv2d) or isinstance(layer, torchlogix.layers.LogicDense):
                        layer.temperature = temperature
                pbar.set_postfix(loss=f"{loss:.4f}", temp=f"{temperature:.4f}")
            
        
            if args.weight_growth != 0.0:
                for param in model.parameters():
                    # draw params closer to 0 or +- 1
                    param.data *= (1 + args.weight_growth)

    # Save final model
    torch.save(model.state_dict(), f"{args.output}/final_model.pt")

    # Save final metrics and config
    save_metrics_csv(metrics, args.output, "training_metrics.csv")

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
        "--architecture", "-a", choices=torchlogix.models.__dict__.keys(),
        default="randomly_connected", help="Model architecture"
    )
    parser.add_argument(
        "--connections", type=str, choices=["random", "unique"],
        default="random", help="Model architecture"
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
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )

    parser.add_argument(
        "--parametrization", type=str, default="raw", choices=["raw", "walsh", "anf", "walsh2"],
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
        help="Temperature for sigmoid in walsh parametrization"
    )

    parser.add_argument(
        "--reg-lambda", type=float, default=0.0,
        help="Regularization strength"
    )

    parser.add_argument(
        "--weight-growth", type=float, default=0.0,
        help="Weight growth factor per iteration"
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