#!/usr/bin/env python3
"""Example: Export a TorchLogix model to Verilog HDL.

This script demonstrates how to create a simple LogicDense network,
train it briefly, and export it to Verilog for comparison with the
C→HLS pipeline.
"""

import torch
import torch.nn as nn
from torchlogix.layers import LogicDense, GroupSum
from torchlogix import CompiledLogicNet


def create_simple_model(input_size=8, hidden_size=4, output_size=2):
    """Create a simple 2-layer LogicDense network."""
    model = nn.Sequential(
        LogicDense(input_size, hidden_size, connections="fixed", device="cpu"),
        LogicDense(hidden_size, hidden_size, connections="fixed", device="cpu"),
        LogicDense(hidden_size, hidden_size, connections="fixed", device="cpu"),
        LogicDense(hidden_size, hidden_size, connections="fixed", device="cpu"),
        LogicDense(hidden_size, hidden_size, connections="fixed", device="cpu"),
        LogicDense(hidden_size, output_size, connections="fixed", device="cpu"),
        GroupSum(1, tau=1.0)
    )
    model.eval()
    return model


def export_to_verilog(model, output_dir="./verilog_output"):
    """Export model to Verilog and C for comparison."""
    print("\nCreating CompiledLogicNet instance...")

    # Create compiled version
    compiled = CompiledLogicNet(model, input_shape=(1,8), use_bitpacking=False, num_bits=1)

    # Generate and save Verilog
    print(f"\nGenerating Verilog...")
    verilog_code = compiled.get_verilog_code(module_name="logic_net")

    # Export to files
    compiled.export_hdl(output_dir, module_name="logic_net", format="verilog")

    # Also generate C code for comparison
    print(f"\nGenerating C code for comparison...")
    c_code = compiled.get_c_code()
    with open(f"{output_dir}/logic_net.c", "w") as f:
        f.write(c_code)
    print(f"C code written to: {output_dir}/logic_net.c")

    # Calculate sizes
    import numpy as np
    layer_info = compiled._calculate_layer_output_sizes_and_shapes()
    if compiled.input_shape is not None:
        input_size = int(np.prod(compiled.input_shape))
    else:
        if len(compiled.linear_layers) > 0:
            input_size = len(compiled.linear_layers[0][0])
        else:
            input_size = 0

    if len(layer_info) > 0:
        output_size = layer_info[-1][3]  # Last layer's output size
    else:
        output_size = 0

    # Print some statistics
    print(f"\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Input size:  {input_size} bits")
    print(f"Output size: {output_size} bits")
    print(f"Linear layers: {len(compiled.linear_layers)}")
    print(f"Conv layers:   {len(compiled.conv_layers)}")
    print(f"\nVerilog lines: {len(verilog_code.split(chr(10)))}")
    print(f"C code lines:  {len(c_code.split(chr(10)))}")
    print("="*60)

    return compiled, verilog_code, c_code


def display_verilog_preview(verilog_code, num_lines=30):
    """Display first few lines of generated Verilog."""
    print(f"\nVerilog Preview (first {num_lines} lines):")
    print("-" * 60)
    lines = verilog_code.split('\n')
    for line in lines[:num_lines]:
        print(line)
    if len(lines) > num_lines:
        print(f"... ({len(lines) - num_lines} more lines)")
    print("-" * 60)


def main():
    print("="*60)
    print("TorchLogix: Verilog Export Example")
    print("="*60)

    # Step 1: Create model
    print("\n[Step 1] Creating model...")
    model = create_simple_model(input_size=8, hidden_size=32_000, output_size=32_000)
    print(f"  Model: {model}")
    print(f"total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Step 2: Export to Verilog
    print("\n[Step 2] Exporting to Verilog...")
    compiled, verilog_code, c_code = export_to_verilog(model, output_dir="./verilog_output")

    # Step 3: Display preview
    display_verilog_preview(verilog_code, num_lines=40)

    # Step 4: Test inference equivalence (optional)
    print("\n[Step 4] Testing inference equivalence...")
    test_input = torch.randint(0, 2, (1, 8)).float()
    print(f"  Test input: {test_input.numpy()}")

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_input)
    print(f"  PyTorch output: {pytorch_output.numpy()}")

    # Compiled inference (using C code)
    print("\nCompiling the model (using C code)...")
    compiled.compile()
    print("\n... done.")
    compiled_output = compiled.forward(test_input.numpy().astype('bool'))
    print(f"  Compiled output: {compiled_output}")

    print("\n" + "="*60)
    print("Done! Files generated in ./verilog_output/")
    print("  - logic_net.v   (Verilog HDL)")
    print("  - logic_net.c   (C code for comparison)")
    print("\nNext steps:")
    print("  - Synthesize with Vivado/Yosys to get resource usage")
    print("  - Use HLS on C code to compare HDL quality")
    print("  - Simulate Verilog to verify functional equivalence")
    print("="*60)


if __name__ == "__main__":
    main()
