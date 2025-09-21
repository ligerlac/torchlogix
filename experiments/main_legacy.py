import csv
import os
import random
import argparse
from typing import Callable, List, Dict

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from results_json import ResultsJSON
from tqdm import tqdm

from torchlogix import CompiledLogicNet, PackBitsTensor
from torchlogix.utils import train

from accuracy_metrics import evaluate_model
from model_selection import get_model
from loading import load_dataset, load_n

import utils

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}

IMPL_TO_DEVICE = {"cuda": "cuda", "python": "cpu"}


def eval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    eval_functions: Dict[str, Callable],
    mode: str = "train",  # train, eval, or packbits
    device="cuda"
):
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


def main(args):

    assert args.num_iterations % args.eval_freq == 0, (
        f"iteration count ({args.num_iterations}) has to be divisible by "
        f"evaluation frequency ({args.eval_freq})"
    )

    device = IMPL_TO_DEVICE[args.implementation]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)

    model, loss_fn, optim = get_model(args)

    model.to(device)

    results = defaultdict(list)
    best_acc = 0

    eval_functions = {
        "loss": loss_fn,
        "acc": lambda preds, y: (preds.argmax(-1) == y).to(torch.float32).mean(),
    }

    modes = ["train", "eval"]
    if args.packbits_eval:
        modes.append("packbits")

    loaders = {"val": validation_loader}
    if args.extensive_eval:
        loaders.update({"train": train_loader, "test": test_loader})

    for i, (x, y) in tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="iteration",
        total=args.num_iterations,
    ):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(
            device
        )
        y = y.to(device)

        loss = train(model, x, y, loss_fn, optim)

        if (i + 1) % args.eval_freq == 0:

            r = {
                "iteration": i + 1,
                "loss": loss,
            }

            metrics = {}
            for loader_name, loader in loaders.items():
                metrics[loader_name] = {}
                for mode in modes:
                    res = eval(model, loader, eval_functions, mode=mode, device=device)
                    metrics[loader_name][mode] = res

            print(metrics)

            if metrics['val']['eval']['acc'] > best_acc:
                best_acc = metrics['val']['eval']['acc']
                print("IS THE BEST UNTIL NOW.")

            # flatten the metrics dict
            for loader_name in metrics.keys():
                for mode in metrics[loader_name].keys():
                    for metric, val in metrics[loader_name][mode].items():
                        r[f"{loader_name}_{mode}_{metric}"] = val

            for key, val in r.items():
                results[key].append(val)

            # write to .csv file
            with open(f"{args.output}/results.csv", 'w') as csv_file:  
                writer = csv.writer(csv_file)
                for key, value in results.items():
                    writer.writerow([key, value])


    # store the model
    torch.save(model.state_dict(), f"{args.output}/model.pt")
    ##########################################################################

    if args.compile_model:
        print("\n" + "=" * 80)
        print(" Converting the model to C code and compiling it...")
        print("=" * 80)

        for opt_level in range(4):
            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs(f"{args.output}/lib", exist_ok=True)
                save_lib_path = f"{args.output}/lib/{opt_level}_{num_bits}.so"

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler="gcc",
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True,
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for data, labels in torch.utils.data.DataLoader(
                        test_loader.dataset, batch_size=int(1e6), shuffle=False
                    ):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print("COMPILED MODEL", num_bits, acc3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train logic gate network on the various datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "adult",
            "breast_cancer",
            "monk1",
            "monk2",
            "monk3",
            "mnist",
            "mnist20x20",
            "cifar-10-3-thresholds",
            "cifar-10-31-thresholds",
            "cora",
            "pubmed",
            "citeseer",
            "nell",
        ],
        required=True,
        help="the dataset to use",
    )
    parser.add_argument(
        "--tau", "-t", type=float, default=10, help="the softmax temperature tau"
    )
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed (default: 0)")
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--training-bit-count",
        "-c",
        type=int,
        default=32,
        help="training bit count (default: 32)",
    )

    parser.add_argument(
        "--implementation",
        type=str,
        default="cuda",
        choices=["cuda", "python"],
        help="`cuda` is the fast CUDA implementation and "
        "`python` is simpler but much slower "
        "implementation intended for helping with the understanding.",
    )

    parser.add_argument(
        "--packbits_eval",
        action="store_true",
        help="Use the PackBitsTensor implementation for an " "additional eval step.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Compile the final model with C for CPU.",
    )

    parser.add_argument(
        "--num-iterations",
        "-ni",
        type=int,
        default=100_000,
        help="Number of iterations (default: 100_000)",
    )
    parser.add_argument(
        "--eval-freq",
        "-ef",
        type=int,
        default=2_000,
        help="Evaluation frequency (default: 2_000)",
    )

    parser.add_argument(
        "--valid-set-size",
        "-vss",
        type=float,
        default=0.2,
        help="Fraction of the train set used for validation (default: 0.2)",
    )
    parser.add_argument(
        "--extensive-eval",
        action="store_true",
        help="Additional evaluation (incl. valid set eval).",
    )

    parser.add_argument(
        "--connections", type=str, default="random", choices=["random", "unique"]
    )
    parser.add_argument("--architecture", "-a", choices=[
        "randomly_connected",
        "fully_connected",
        "cnn",
    ], default="randomly_connected")
    parser.add_argument("--num_neurons", "-k", type=int, default=6000)
    parser.add_argument("--num_layers", "-l", type=int, default=4)

    parser.add_argument("--grad-factor", type=float, default=1.0)

    parser.add_argument(
        "--output", "-o",
        action=utils.CreateFolder,
        type=Path,
        default="results/latest/",
        help="Path to directory where results will be stored",
    )

    main(parser.parse_args())
