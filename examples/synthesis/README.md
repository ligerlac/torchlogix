# TorchLogix Verilog Synthesis Examples

This directory contains tools and examples for synthesizing and testing TorchLogix-generated Verilog on FPGAs.

## Quick Start

### 1. Generate Verilog from Your Model

```bash
cd ../  # Go to examples directory
python export_to_verilog.py  # or export_conv_to_verilog.py
```

This creates a `verilog_output/` directory with your Verilog module.

### 2. Generate Test Vectors

```bash
cd synthesis/
python generate_test_vectors.py --input-size 8 --num-tests 100 --output-dir ../verilog_output/
```

### 3. Run Functional Testing

```bash
cd ../verilog_output/
cp ../synthesis/testbench_template.v tb_logic_net.v

# Edit tb_logic_net.v to match your module's INPUT_WIDTH and OUTPUT_WIDTH

# Simulate with Icarus Verilog
iverilog -o sim.out logic_net.v tb_logic_net.v
vvp sim.out
```

### 4. Run FPGA Synthesis

```bash
# Make sure Vivado is in your PATH
vivado -mode batch -source ../synthesis/synthesize.tcl \
    -tclargs logic_net.v xc7z020clg400-1 synthesis_reports/

# View results
cat synthesis_reports/summary.txt
```

## Files in This Directory

| File | Description |
|------|-------------|
| `synthesize.tcl` | Automated Vivado synthesis script |
| `testbench_template.v` | Verilog testbench template |
| `generate_test_vectors.py` | Python script to generate test vectors |
| `README.md` | This file |

## Complete Workflow Example

Here's a complete example workflow from model to synthesis:

```bash
# 1. Start in the examples directory
cd /path/to/torchlogix/examples/

# 2. Export a model to Verilog
python export_to_verilog.py
# Output: verilog_output/logic_net.v

# 3. Generate test vectors
cd synthesis/
python generate_test_vectors.py \
    --input-size 8 \
    --num-tests 100 \
    --output-dir ../verilog_output/

# 4. Copy and configure testbench
cp testbench_template.v ../verilog_output/tb_logic_net.v
# Edit tb_logic_net.v if needed (INPUT_WIDTH, OUTPUT_WIDTH)

# 5. Run functional simulation
cd ../verilog_output/
iverilog -o sim.out logic_net.v tb_logic_net.v
vvp sim.out

# Expected output:
# ========================================
# TorchLogix Verilog Testbench
# ========================================
# ...
# RESULT: ALL TESTS PASSED
# ========================================

# 6. Run FPGA synthesis (requires Vivado)
vivado -mode batch \
    -source ../synthesis/synthesize.tcl \
    -tclargs logic_net.v xc7z020clg400-1 synthesis_reports/

# 7. View synthesis results
cat synthesis_reports/summary.txt
```

## Synthesis Script Usage

The `synthesize.tcl` script accepts the following arguments:

```bash
vivado -mode batch -source synthesize.tcl -tclargs <verilog_file> <part> [output_dir] [clock_period]
```

**Arguments:**
- `verilog_file`: Path to Verilog file (required)
- `part`: Target FPGA part number (required)
- `output_dir`: Directory for reports (optional, default: synthesis_reports)
- `clock_period`: Target clock period in ns (optional, default: 10.0)

**Examples:**

```bash
# Basic synthesis for Pynq-Z2
vivado -mode batch -source synthesize.tcl \
    -tclargs logic_net.v xc7z020clg400-1

# Custom output directory
vivado -mode batch -source synthesize.tcl \
    -tclargs logic_net.v xc7z020clg400-1 my_reports/

# Target 5ns clock period (200 MHz)
vivado -mode batch -source synthesize.tcl \
    -tclargs logic_net.v xc7z020clg400-1 reports/ 5.0
```

## Common FPGA Parts

| Board | Part Number | Family |
|-------|-------------|--------|
| Pynq-Z2 | `xc7z020clg400-1` | Zynq-7000 |
| ZCU104 | `xczu7ev-ffvc1156-2-e` | Zynq UltraScale+ |
| Arty A7-35T | `xc7a35tcpg236-1` | Artix-7 |
| Arty A7-100T | `xc7a100tcsg324-1` | Artix-7 |
| KCU116 | `xcku5p-ffvb676-2-e` | Kintex UltraScale+ |
| VC707 | `xc7vx485tffg1761-2` | Virtex-7 |

## Understanding Synthesis Results

After synthesis completes, check `synthesis_reports/summary.txt`:

```
=========================================
Synthesis Summary
=========================================
Design:          logic_net
Target part:     xc7z020clg400-1
Clock period:    10.0 ns
Synthesis time:  23 seconds

LUTs:            147
Flip-Flops:      0
DSPs:            0
BRAM Tiles:      0

WNS:             7.5 ns
Achieved fmax:   400.00 MHz (based on 10.0ns target)
Critical path:   2.5 ns
Latency:         2.5 ns (1 clock cycle)

Total power:     0.082 W
Dynamic power:   0.012 W
Static power:    0.070 W

=========================================
```

**Key Metrics Explained:**

- **LUTs**: Number of Look-Up Tables used (primary logic resource)
- **Flip-Flops**: Number of registers (typically 0 for TorchLogix combinational logic)
- **WNS**: Worst Negative Slack - positive means timing is met
- **Critical Path**: Longest combinational delay (this is your latency!)
- **Achieved fmax**: Maximum frequency if used in a clocked design

## Test Vector Generation

The `generate_test_vectors.py` script supports several modes:

```bash
# Generate vectors for a specific input size
python generate_test_vectors.py --input-size 8 --num-tests 100

# Load a saved model
python generate_test_vectors.py --model ../models/trained_model.pt --num-tests 1000

# For convolutional models with 8x8 input
python generate_test_vectors.py --input-shape 8 8 --num-tests 500

# Use a specific random seed for reproducibility
python generate_test_vectors.py --input-size 8 --num-tests 100 --seed 42
```

## Troubleshooting

### Simulation Failures

**Problem:** Testbench reports mismatched outputs
- **Solution:** Verify test vectors were generated from the same model version
- Check INPUT_WIDTH and OUTPUT_WIDTH in testbench match your module

**Problem:** Output contains 'X' or 'Z' values
- **Solution:** Check that all wires in Verilog are assigned
- Verify no undriven nets in generated Verilog

### Synthesis Failures

**Problem:** "Multi-driven net" error
- **Solution:** Check for duplicate wire assignments in generated Verilog
- Verify only one `assign` statement per output wire

**Problem:** Very high LUT count
- **Solution:** Check model complexity - may need to simplify architecture
- Verify gate operations are optimal

**Problem:** Timing not met (negative WNS)
- **Solution:** This usually means your critical path is longer than the clock period
- For combinational logic, this is informational - check critical path delay instead

## Integration with Python

You can automate the entire flow from Python:

```python
import subprocess
from pathlib import Path

def synthesize_and_test(model, verilog_file, part='xc7z020clg400-1'):
    """Complete synthesis and testing workflow."""

    # 1. Generate test vectors
    subprocess.run([
        'python', 'generate_test_vectors.py',
        '--model', str(model),
        '--num-tests', '100'
    ])

    # 2. Run simulation
    subprocess.run(['iverilog', '-o', 'sim.out', verilog_file, 'tb_logic_net.v'])
    result = subprocess.run(['vvp', 'sim.out'], capture_output=True, text=True)

    if 'ALL TESTS PASSED' not in result.stdout:
        raise RuntimeError('Simulation failed')

    # 3. Run synthesis
    subprocess.run([
        'vivado', '-mode', 'batch',
        '-source', 'synthesize.tcl',
        '-tclargs', verilog_file, part
    ])

    # 4. Parse results
    with open('synthesis_reports/summary.txt') as f:
        summary = f.read()

    return summary

# Example usage
summary = synthesize_and_test('model.pt', 'logic_net.v')
print(summary)
```

## Next Steps

- **Full Implementation**: Run place & route for final timing numbers
- **Bitstream Generation**: Create FPGA configuration file and test on hardware
- **Performance Optimization**: Try different synthesis directives for area/speed tradeoffs

## Documentation

For more detailed information, see:
- [Verilog Testing Guide](../../docs/verilog_testing.md)
- [Vivado Synthesis Guide](../../docs/vivado_synthesis.md)
- [TorchLogix Documentation](../../README.md)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the detailed guides in `docs/`
3. Verify Vivado version compatibility (tested with 2019.1+)
4. Ensure test vectors match your Verilog module specifications
