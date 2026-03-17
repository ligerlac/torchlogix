#!/usr/bin/env python3
"""Example: Export TorchLogix models with different pipeline configurations.

This script demonstrates how to generate pipelined Verilog to improve
synthesis time and achieve better timing for large models.

Pipeline strategies:
- pipeline_stages=0: Fully combinational (default, may fail for large models)
- pipeline_stages=1: Single output register (helps synthesis)
- pipeline_stages=N: N pipeline stages (N cycles latency, better fmax)
"""

import torch
import torch.nn as nn
from torchlogix.layers import LogicDense, GroupSum
from torchlogix import CompiledLogicNet
import os


def create_large_model(input_size=16, num_layers=8, hidden_size=32):
    """Create a larger multi-layer LogicDense network."""
    layers = []

    # First layer
    layers.append(LogicDense(input_size, hidden_size, connections="fixed", device="cpu"))

    # Hidden layers
    for i in range(num_layers - 2):
        layers.append(LogicDense(hidden_size, hidden_size, connections="fixed", device="cpu"))

    # Output layer
    layers.append(LogicDense(hidden_size, 16, connections="fixed", device="cpu"))

    # Group Sum
    layers.append(GroupSum(1, tau=1.0))

    model = nn.Sequential(*layers)
    return model


def export_with_pipelining(model, pipeline_stages, output_dir):
    """Export model with specified pipelining."""
    print(f"\n{'='*70}")
    print(f"Pipeline stages: {pipeline_stages}")
    print(f"{'='*70}")

    # Create compiled version
    compiled = CompiledLogicNet(model, input_shape=(1, 16))

    # Generate Verilog with pipelining
    module_name = f"logic_net_p{pipeline_stages}"
    verilog_code = compiled.get_verilog_code(
        module_name=module_name,
        pipeline_stages=pipeline_stages
    )

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{module_name}.v"
    with open(filename, 'w') as f:
        f.write(verilog_code)

    # Print statistics
    lines = verilog_code.split('\n')
    num_regs = sum(1 for line in lines if 'reg [' in line and 'layer' in line)
    num_wires = sum(1 for line in lines if 'wire [' in line)
    has_clock = 'input wire clk' in verilog_code

    print(f"Output: {filename}")
    print(f"Module: {module_name}")
    print(f"Clocked design: {'Yes' if has_clock else 'No'}")
    print(f"Registers: {num_regs}")
    print(f"Wires: {num_wires}")
    print(f"Verilog lines: {len(lines)}")

    # Show first few lines
    print(f"\nModule header:")
    for line in lines[:15]:
        if line.strip():
            print(f"  {line}")

    return verilog_code, num_regs


def main():
    print("="*70)
    print("TorchLogix: Pipelined Verilog Export Example")
    print("="*70)

    # Create a larger model to demonstrate pipelining benefits
    print("\n[Step 1] Creating multi-layer model...")
    input_size = 16
    num_layers = 8
    hidden_size = 32

    model = create_large_model(input_size, num_layers, hidden_size)
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Model: {len(list(model.modules()))-1} layers")  # -1 for Sequential container

    output_dir = "./pipelined_verilog_output"

    # Export with different pipeline configurations
    print(f"\n[Step 2] Exporting with different pipeline configurations...")

    configs = [
        (0, "Fully combinational - may be slow to synthesize for large models"),
        (1, "Single output register - helps with large combinational paths"),
        (2, "2 pipeline stages - half the layers per stage"),
        (4, "4 pipeline stages - better fmax, 4 cycle latency"),
        (num_layers, "Full layer-level pipelining - highest fmax, 1 layer per cycle"),
    ]

    results = []
    for pipeline_stages, description in configs:
        print(f"\n{description}")
        verilog, num_regs = export_with_pipelining(model, pipeline_stages, output_dir)
        results.append((pipeline_stages, num_regs, len(verilog.split('\n'))))

    # Summary table
    print(f"\n{'='*70}")
    print("Summary of Pipeline Configurations")
    print(f"{'='*70}")
    print(f"{'Stages':<8} {'Latency':<10} {'Registers':<12} {'Verilog Lines':<15} {'Best For':<20}")
    print(f"{'-'*70}")

    use_cases = [
        "Small models, max speed",
        "Large models, low latency",
        "Balanced",
        "Higher fmax",
        "Maximum fmax"
    ]

    for (stages, regs, lines), use_case in zip(results, use_cases):
        latency = f"{stages} cycle{'s' if stages != 1 else ''}"
        print(f"{stages:<8} {latency:<10} {regs:<12} {lines:<15} {use_case:<20}")

    print(f"{'='*70}")

    # Performance characteristics
    print("\n[Step 3] Performance Characteristics")
    print(f"{'='*70}")
    print("\nCombinational vs Pipelined Trade-offs:")
    print("\nCombinational (pipeline_stages=0):")
    print("  ✓ Lowest latency (1 cycle)")
    print("  ✗ May fail synthesis for large models (>1M Verilog lines)")
    print("  ✗ Lower fmax (long critical paths)")
    print("  ✗ Very slow synthesis time")
    print("\nPipelined (pipeline_stages>0):")
    print("  ✓ Synthesis succeeds even for large models")
    print("  ✓ Higher fmax (shorter critical paths)")
    print("  ✓ Faster synthesis time")
    print("  ✓ Predictable timing closure")
    print("  ✗ Higher latency (N cycles)")
    print("  ✗ More area (registers)")

    print("\n[Step 4] Recommended Configurations")
    print(f"{'='*70}")
    print("\nModel Size Guide:")
    print("  • Small (<10 layers):     pipeline_stages=0  (fully combinational)")
    print("  • Medium (10-50 layers):  pipeline_stages=1-4")
    print("  • Large (50-200 layers):  pipeline_stages=4-16")
    print("  • Very Large (>200):      pipeline_stages=N/4 (N=num_layers)")
    print("\nSynthesis Issues?")
    print("  If synthesis fails or is very slow:")
    print("  1. Start with pipeline_stages=1 (just output register)")
    print("  2. If still slow, increase to 2, 4, 8, etc.")
    print("  3. For fastest synthesis: use full layer pipelining")

    print("\n[Step 5] Next Steps")
    print(f"{'='*70}")
    print(f"Files generated in: {output_dir}/")
    print("\nTo synthesize and compare:")
    for stages, _, _ in results:
        module_name = f"logic_net_p{stages}"
        print(f"\n  # Pipeline stages = {stages}")
        print(f"  vivado -mode batch -source ../synthesis/synthesize.tcl \\")
        print(f"    -tclargs {output_dir}/{module_name}.v xc7z020clg400-1 \\")
        print(f"    {output_dir}/reports_p{stages}/")

    print("\nThen compare:")
    print("  • synthesis_reports/summary.txt - resource usage and timing")
    print("  • Look for: LUTs, FFs, WNS, fmax, critical path")
    print("  • Pipelined designs should have higher fmax, more FFs")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
