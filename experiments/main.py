import os
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from results_json import ResultsJSON
from tqdm import tqdm

from torchlogix import CompiledLogicNet
from torchlogix.utils import eval, packbits_eval, train

from accuracy_metrics import evaluate_model
from model_selection import get_model
from loading import load_dataset, load_n

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}

IMPL_TO_DEVICE = {"cuda": "cuda", "python": "cpu" ""}


def main(args):

    assert args.num_iterations % args.eval_freq == 0, (
        f"iteration count ({args.num_iterations}) has to be divisible by "
        f"evaluation frequency ({args.eval_freq})"
    )

    # try:
    #     import difflogic_cuda
    # except ImportError:
    #     warnings.warn('failed to import difflogic_cuda.
    # no cuda features will be available', ImportWarning)

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path="./results/")
        results.store_args(args)

    device = IMPL_TO_DEVICE[args.implementation]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)

    model.to(device)

    best_acc = 0
    if test_loader is not None:
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
                if args.extensive_eval:
                    train_accuracy_train_mode = eval(
                        model, train_loader, mode=True, device=device
                    )
                    valid_accuracy_eval_mode = eval(
                        model, validation_loader, mode=False, device=device
                    )
                    valid_accuracy_train_mode = eval(
                        model, validation_loader, mode=True, device=device
                    )
                else:
                    train_accuracy_train_mode = -1
                    valid_accuracy_eval_mode = -1
                    valid_accuracy_train_mode = -1
                train_accuracy_eval_mode = eval(
                    model, train_loader, mode=False, device=device
                )
                test_accuracy_eval_mode = eval(
                    model, test_loader, mode=False, device=device
                )
                test_accuracy_train_mode = eval(
                    model, test_loader, mode=True, device=device
                )

                r = {
                    "train_acc_eval_mode": train_accuracy_eval_mode,
                    "train_acc_train_mode": train_accuracy_train_mode,
                    "valid_acc_eval_mode": valid_accuracy_eval_mode,
                    "valid_acc_train_mode": valid_accuracy_train_mode,
                    "test_acc_eval_mode": test_accuracy_eval_mode,
                    "test_acc_train_mode": test_accuracy_train_mode,
                }

                if args.packbits_eval:
                    r["train_acc_eval"] = packbits_eval(
                        model, train_loader, device=device
                    )
                    r["valid_acc_eval"] = packbits_eval(
                        model, train_loader, device=device
                    )
                    r["test_acc_eval"] = packbits_eval(
                        model, test_loader, device=device
                    )

                if args.experiment_id is not None:
                    results.store_results(r)

                if valid_accuracy_eval_mode > best_acc:
                    best_acc = valid_accuracy_eval_mode
                    if args.experiment_id is not None:
                        results.store_final_results(r)
                    else:
                        print("IS THE BEST UNTIL NOW.")

                if args.experiment_id is not None:
                    results.save()
    else:
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=5e-4
        )
        data = train_loader.to(device)
        # data.x = data.x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE
        # [args.training_bit_count]).to(device)

        for epoch in range(100000):
            optimizer.zero_grad()
            out = model(data)
            if args.architecture in ["gcn_cora_baseline"]:
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            else:
                loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Gradient for {name} is None")

            optimizer.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())
                val_metrics = evaluate_model(
                    model, data, data.test_mask, int(max(data.y) + 1)
                )
                print(val_metrics["accuracy"], val_metrics["confusion_matrix"])
                model.train()
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
                os.makedirs("lib", exist_ok=True)
                save_lib_path = "lib/{:08d}_{}.so".format(
                    args.experiment_id if args.experiment_id is not None else 0,
                    num_bits,
                )

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

    parser.add_argument("-eid", "--experiment_id", type=int, default=None)

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
        default=0.0,
        help="Fraction of the train set used for validation (default: 0.)",
    )
    parser.add_argument(
        "--extensive-eval",
        action="store_true",
        help="Additional evaluation (incl. valid set eval).",
    )

    parser.add_argument(
        "--connections", type=str, default="random", choices=["random", "unique"]
    )
    # parser.add_argument("--architecture", "-a", type=str, default="randomly_connected")
    parser.add_argument("--architecture", "-a", choices=[
        "randomly_connected",
        "fully_connected",
        "cnn",
    ], default="randomly_connected")
    parser.add_argument("--num_neurons", "-k", type=int, default=6000)
    parser.add_argument("--num_layers", "-l", type=int, default=4)

    parser.add_argument("--grad-factor", type=float, default=1.0)

    main(parser.parse_args())
