#!/usr/bin/env python3
"""Generate test vectors for a TorchLogix Circuit.

Compiles a trained model to a Circuit, runs random boolean inputs through it,
and writes the results in $readmemb-compatible text format for use with the
Verilog testbench template.

Usage:
    # Load a saved model (must also specify --input-shape)
    python generate_test_vectors.py --model model.pt --input-shape 1 28 28 --num-tests 100

    # Generate vectors for a small demo circuit (no model file needed)
    python generate_test_vectors.py --input-size 8 --num-tests 100
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def build_circuit(model, input_shape):
    from torchlogix import Circuit
    from torchlogix.utils import set_export_mode

    model.eval()
    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=input_shape)
    circuit.simplify()
    return circuit


def generate_test_vectors(circuit, num_tests: int, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_in = circuit.n_inputs
    print(f"Circuit inputs:  {n_in} bits")
    print(f"Generating {num_tests} test vectors...")

    rng = np.random.default_rng(42)
    inputs = rng.integers(0, 2, (num_tests, n_in), dtype=np.uint8)

    outputs = []
    for i, row in enumerate(inputs):
        if (i + 1) % max(1, num_tests // 10) == 0:
            print(f"  {i + 1}/{num_tests}", end="\r")
        out = circuit(row.reshape(1, -1).astype(bool))
        outputs.append(out[0])
    print(f"  {num_tests}/{num_tests} done")

    outputs = np.array(outputs)

    input_file = output_dir / "test_vectors_input.txt"
    output_file = output_dir / "test_vectors_output.txt"

    with open(input_file, "w") as f:
        for row in inputs:
            f.write("".join(str(b) for b in row) + "\n")

    # outputs may be bool or integer; write as binary for bool, decimal for int
    is_bool = outputs.dtype == bool or (outputs.ndim == 2 and outputs.max() <= 1)
    with open(output_file, "w") as f:
        for row in outputs:
            if is_bool or outputs.max() <= 1:
                f.write("".join(str(int(b)) for b in row) + "\n")
            else:
                # GroupSum outputs — write space-separated integers
                f.write(" ".join(str(int(v)) for v in row) + "\n")

    np.savez(output_dir / "test_vectors.npz", inputs=inputs, outputs=outputs)

    print(f"\nWrote {num_tests} vectors to {output_dir}/")
    print(f"  {input_file.name}  — $readmemb input file")
    print(f"  {output_file.name} — $readmemb output file")
    print(f"  test_vectors.npz   — NumPy archive for debugging")

    print(f"\nSample (first 5):")
    print(f"  {'Input':<{min(n_in, 20)}}  Output")
    for i in range(min(5, num_tests)):
        inp_s = "".join(str(b) for b in inputs[i])[:20]
        out_s = "".join(str(int(b)) for b in outputs[i]) if is_bool else str(outputs[i])
        print(f"  {inp_s}  {out_s}")
    if num_tests > 5:
        print(f"  ... ({num_tests - 5} more)")

    return inputs, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Verilog test vectors from a TorchLogix Circuit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--model", type=str,
                             help="Path to a saved model (.pt) — also requires --input-shape")
    input_group.add_argument("--input-size", type=int,
                             help="Build a small demo LogicDense circuit with this many inputs")

    parser.add_argument("--input-shape", type=int, nargs="+",
                        help="Input shape for --model (e.g. 1 28 28 for MNIST)")
    parser.add_argument("--num-tests", type=int, default=100,
                        help="Number of test vectors to generate (default: 100)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory (default: current directory)")

    args = parser.parse_args()

    if args.model:
        if not args.input_shape:
            parser.error("--model requires --input-shape")
        print(f"Loading model: {args.model}")
        model = torch.load(args.model, map_location="cpu", weights_only=False)
        input_shape = tuple(args.input_shape)
    else:
        n = args.input_size
        print(f"Building demo LogicDense circuit ({n} → {max(1, n // 2)} → {max(1, n // 4)})")
        from torchlogix.layers import LogicDense
        model = nn.Sequential(
            LogicDense(n, max(1, n // 2)),
            LogicDense(max(1, n // 2), max(1, n // 4)),
        )
        input_shape = (n,)

    circuit = build_circuit(model, input_shape)
    generate_test_vectors(circuit, args.num_tests, args.output_dir)


if __name__ == "__main__":
    main()
