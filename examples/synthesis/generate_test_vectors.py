#!/usr/bin/env python3
"""Generate test vectors for Verilog testbench.

This script generates random binary test inputs and computes expected outputs
using a trained TorchLogix model. The test vectors are saved in a format
compatible with the Verilog testbench template.

Usage:
    python generate_test_vectors.py --model model.pt --num-tests 100 --output test_vectors/
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys


def generate_test_vectors(model, num_tests: int, input_shape, output_dir: Path):
    """Generate test vectors for a TorchLogix model.

    Args:
        model: Trained TorchLogix model
        num_tests: Number of test cases to generate
        input_shape: Shape of input tensor (without batch dimension)
        output_dir: Directory to save test vectors
    """
    from torchlogix import CompiledLogicNet

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_tests} test vectors...")
    print(f"Input shape: {input_shape}")

    # Compile model
    compiled = CompiledLogicNet(model)
    compiled.compile()

    # Determine input size
    if isinstance(input_shape, tuple):
        input_size = int(np.prod(input_shape))
    else:
        input_size = input_shape

    print(f"Input size: {input_size} bits")

    # Generate random binary inputs
    test_inputs = np.random.randint(0, 2, (num_tests, input_size), dtype=np.int8)

    # Get expected outputs
    print("Computing expected outputs...")
    test_outputs = []
    for i, inp in enumerate(test_inputs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing test {i+1}/{num_tests}...", end='\r')

        # Reshape if needed
        if isinstance(input_shape, tuple):
            inp_shaped = inp.reshape(1, *input_shape)
        else:
            inp_shaped = inp.reshape(1, -1)

        out = compiled.forward(inp_shaped)
        test_outputs.append(out[0])

    print(f"  Processing test {num_tests}/{num_tests}... Done!")

    test_outputs = np.array(test_outputs)
    output_size = test_outputs.shape[1]

    print(f"Output size: {output_size} bits")

    # Save to files (binary format for Verilog $readmemb)
    input_file = output_dir / "test_vectors_input.txt"
    output_file = output_dir / "test_vectors_output.txt"

    print(f"\nSaving test vectors:")
    print(f"  Inputs:  {input_file}")
    print(f"  Outputs: {output_file}")

    # Write in binary format
    with open(input_file, 'w') as f:
        for inp in test_inputs:
            # Convert to binary string
            binary = ''.join(str(bit) for bit in inp)
            f.write(binary + '\n')

    with open(output_file, 'w') as f:
        for out in test_outputs:
            binary = ''.join(str(int(bit)) for bit in out)
            f.write(binary + '\n')

    # Also save in NumPy format for debugging
    np.savez(output_dir / "test_vectors.npz",
             inputs=test_inputs,
             outputs=test_outputs)

    print(f"\nTest vector generation complete!")
    print(f"Use these files with the Verilog testbench template.")

    # Print some example vectors
    print(f"\nExample test vectors:")
    print(f"{'Input':<20} | Output")
    print("-" * 30)
    for i in range(min(5, num_tests)):
        inp_str = ''.join(str(bit) for bit in test_inputs[i])
        out_str = ''.join(str(int(bit)) for bit in test_outputs[i])
        print(f"{inp_str:<20} | {out_str}")

    if num_tests > 5:
        print(f"... ({num_tests - 5} more)")

    return test_inputs, test_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate test vectors for TorchLogix Verilog testbench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 test vectors for an 8-input model
  python generate_test_vectors.py --input-size 8 --num-tests 100

  # Load a saved model and generate vectors
  python generate_test_vectors.py --model model.pt --num-tests 1000

  # For a convolutional model with specific input shape
  python generate_test_vectors.py --input-shape 8 8 --num-tests 500
        """
    )

    # Input specification (one of these is required)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--model', type=str,
                            help='Path to saved TorchLogix model (.pt file)')
    input_group.add_argument('--input-size', type=int,
                            help='Input size in bits (for flat input)')
    input_group.add_argument('--input-shape', type=int, nargs='+',
                            help='Input shape (for convolutional models, e.g., 8 8 for 8x8)')

    parser.add_argument('--num-tests', type=int, default=100,
                       help='Number of test vectors to generate (default: 100)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for test vectors (default: current directory)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Determine how to get the model
    if args.model:
        print(f"Loading model from: {args.model}")
        model = torch.load(args.model)
        model.eval()

        # Try to infer input shape
        # This is a heuristic - may need adjustment for your model
        try:
            from torchlogix import CompiledLogicNet
            compiled = CompiledLogicNet(model)
            if compiled.input_shape is not None:
                input_shape = compiled.input_shape
            else:
                # Fallback: try first layer
                first_layer = next(model.modules())
                if hasattr(first_layer, 'in_features'):
                    input_shape = first_layer.in_features
                elif hasattr(first_layer, 'in_dim'):
                    input_shape = first_layer.in_dim
                else:
                    print("ERROR: Could not infer input shape from model")
                    print("Please specify --input-size or --input-shape")
                    sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to infer input shape: {e}")
            print("Please specify --input-size or --input-shape")
            sys.exit(1)

    elif args.input_size:
        print(f"Creating dummy model with input size: {args.input_size}")
        # Create a simple identity model for testing
        from torchlogix.layers import LogicDense
        model = nn.Sequential(
            LogicDense(args.input_size, args.input_size, connections="random", device="cpu")
        )
        input_shape = args.input_size

    elif args.input_shape:
        print(f"Creating dummy model with input shape: {args.input_shape}")
        from torchlogix.layers import LogicConv2d
        if len(args.input_shape) == 2:
            h, w = args.input_shape
            input_shape = (h, w)
            # Create a simple conv model
            model = nn.Sequential(
                LogicConv2d(in_dim=(h, w), channels=1, num_kernels=1,
                           receptive_field_size=3, tree_depth=1,
                           connections="random", device="cpu"),
                nn.Flatten()
            )
        else:
            print("ERROR: --input-shape must have 2 values (height, width)")
            sys.exit(1)

    else:
        print("ERROR: Must specify one of --model, --input-size, or --input-shape")
        parser.print_help()
        sys.exit(1)

    # Generate test vectors
    generate_test_vectors(model, args.num_tests, input_shape, args.output_dir)


if __name__ == '__main__':
    main()
