#!/usr/bin/env python3
"""Example: Export a TorchLogix convolutional model to Verilog HDL.

This script demonstrates Verilog generation for LogicConv2d layers,
showcasing how the binary tree structure is translated to HDL.
"""

import torch
import torch.nn as nn
from torchlogix.layers import LogicConv2d, LogicDense, GroupSum
from torchlogix import CompiledLogicNet


def create_conv_model():
    """Create a simple convolutional model.

    Architecture:
    - Input: 1 channel, 8x8 image
    - Conv1: 2 kernels, 3x3 receptive field, tree_depth=2
    - Flatten
    - Dense: 4 outputs
    """
    model = nn.Sequential(
        LogicConv2d(
            in_dim=(16, 16),           # Input dimensions (H, W)
            channels=1,               # Input channels
            num_kernels=8,            # Output channels
            receptive_field_size=3,   # 3x3 kernel
            tree_depth=3,
            connections="fixed",
            stride=1,
            padding=0,
            device="cpu"              # Use CPU (no CUDA required)
        ),
        nn.Flatten(),
        LogicDense(14*14*8, 8_000, connections="fixed", device="cpu"),  # 2 * 3 * 3 = 18 after conv
#        LogicDense(8_000, 8_000, connections="fixed", device="cpu"),
#        LogicDense(8_000, 8_000, connections="fixed", device="cpu"),
#        LogicDense(8_000, 8_000, connections="fixed", device="cpu"),
        GroupSum(1, tau=1.0)
    )
    return model


def export_to_verilog(model, output_dir="./conv_verilog_output"):
    """Export convolutional model to Verilog."""
    import numpy as np
    import os

    print("\nCompiling model...")
    compiled = CompiledLogicNet(model, input_shape=(1, 16, 16), use_bitpacking=False, num_bits=1)

    # Generate Verilog
    print(f"\nGenerating Verilog...")
    verilog_code = compiled.get_verilog_code(module_name="conv_logic_net")

    # Export to files
    os.makedirs(output_dir, exist_ok=True)
    compiled.export_hdl(output_dir, module_name="conv_logic_net", format="verilog")

    # Also generate C code for comparison
    print(f"\nGenerating C code for comparison...")
    c_code = compiled.get_c_code()
    with open(f"{output_dir}/conv_logic_net.c", "w") as f:
        f.write(c_code)
    print(f"C code written to: {output_dir}/conv_logic_net.c")

    # Calculate sizes
    layer_info = compiled._calculate_layer_output_sizes_and_shapes()
    if compiled.input_shape is not None:
        input_size = int(np.prod(compiled.input_shape))
    else:
        input_size = 0

    if len(layer_info) > 0:
        output_size = layer_info[-1][3]
    else:
        output_size = 0

    # Print statistics
    print(f"\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Model architecture:")
    for i, (layer_type, layer_idx, output_shape, output_size_) in enumerate(layer_info):
        print(f"  Layer {i} ({layer_type}): output_shape={output_shape}, size={output_size_}")
    print(f"\nInput size:  {input_size} bits")
    print(f"Output size: {output_size} bits")
    print(f"Linear layers: {len(compiled.linear_layers)}")
    print(f"Conv layers:   {len(compiled.conv_layers)}")

    if len(compiled.conv_layers) > 0:
        conv_info = compiled.conv_layers[0]
        print(f"\nConv layer details:")
        print(f"  Kernels: {conv_info['num_kernels']}")
        print(f"  Tree depth: {conv_info['tree_depth']}")
        print(f"  Receptive field: {conv_info['receptive_field_size']}")
        print(f"  Input dimension: {conv_info['in_dim']}")

    print(f"\nGenerated code:")
    print(f"  Verilog lines: {len(verilog_code.split(chr(10)))}")
    print(f"  C code lines:  {len(c_code.split(chr(10)))}")
    print("="*60)

    return compiled, verilog_code, c_code


def display_verilog_structure(verilog_code):
    """Display structure of generated Verilog."""
    print(f"\nVerilog Structure Analysis:")
    print("-" * 60)

    lines = verilog_code.split('\n')

    # Count different elements
    wire_count = sum(1 for line in lines if 'wire' in line and '=' not in line)
    assign_count = sum(1 for line in lines if 'assign' in line)
    conv_count = sum(1 for line in lines if 'conv_l' in line and 'wire' in line)

    print(f"Total lines: {len(lines)}")
    print(f"Wire declarations: {wire_count}")
    print(f"Assignments: {assign_count}")
    print(f"Convolutional wires: {conv_count}")

    # Show first 60 lines
    print(f"\nFirst 60 lines of Verilog:")
    print("-" * 60)
    for i, line in enumerate(lines[:60]):
        print(f"{i+1:3d}: {line}")

    if len(lines) > 60:
        print(f"\n... ({len(lines) - 60} more lines)")
    print("-" * 60)


def main():
    print("="*60)
    print("TorchLogix: Convolutional Model Verilog Export Example")
    print("="*60)

    # Step 1: Create model
    print("\n[Step 1] Creating convolutional model...")
    model = create_conv_model()
    print(f"  Model: {model}")
    print(f"total number of parameters: {sum(p.numel() for p in model.parameters())}")


    # Step 3: Export to Verilog
    print("\n[Step 3] Exporting to Verilog...")
    compiled, verilog_code, c_code = export_to_verilog(model, output_dir="./conv_verilog_output")

    # Step 4: Analyze structure
    print("\n[Step 4] Analyzing Verilog structure...")
    display_verilog_structure(verilog_code)

    # Step 5: Test inference
    print("\n[Step 5] Testing inference...")
    test_input = torch.randint(0, 2, (1, 1, 16, 16)).float()
    print(f"  Test input shape: {test_input.shape}")

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_input)
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  PyTorch output: {pytorch_output.numpy()}")

    # Compiled inference
    try:
        compiled.compile()
        test_input_flat = test_input.squeeze(0).numpy().astype(bool)
        compiled_output = compiled.forward(test_input_flat)
        print(f"  Compiled output: {compiled_output}")
    except Exception as e:
        print(f"  Compiled inference skipped: {e}")

    print("\n" + "="*60)
    print("Done! Files generated in ./conv_verilog_output/")
    print("  - conv_logic_net.v   (Verilog HDL)")
    print("  - conv_logic_net.c   (C code for comparison)")
    print("\nKey observations:")
    print("  - Binary tree structure creates hierarchical wire assignments")
    print("  - Each kernel position generates multiple intermediate wires")
    print("  - Gate operations combine inputs through tree levels")
    print("  - Final level outputs to the layer output bus")
    print("="*60)


if __name__ == "__main__":
    main()
