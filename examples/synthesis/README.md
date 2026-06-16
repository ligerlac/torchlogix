# TorchLogix Verilog Synthesis

This directory contains tools for synthesizing and testing TorchLogix-generated
Verilog on FPGAs.

## Workflow

### 1. Export Verilog from a trained model

```python
from torchlogix import Circuit
from torchlogix.utils import set_export_mode

set_export_mode(model)
circuit = Circuit.from_model(model, input_shape=(1, 28, 28))
circuit.simplify()
circuit.write_verilog_code("circuit.v")
```

### 2. Generate test vectors

```bash
# From a saved model
python examples/synthesis/generate_test_vectors.py \
    --model trained_model.pt --input-shape 1 28 28 \
    --num-tests 1000 --output-dir test_vectors/

# Quick demo with a small generated circuit
python examples/synthesis/generate_test_vectors.py \
    --input-size 8 --num-tests 100 --output-dir test_vectors/
```

Writes `test_vectors_input.txt` and `test_vectors_output.txt` in `$readmemb`
format, plus a `test_vectors.npz` NumPy archive for debugging.

### 3. Functional simulation

For a self-contained simulation that requires no manual testbench editing, use
`examples/verify_with_verilator.py` (requires Verilator):

```bash
python examples/verify_with_verilator.py
# → PASS: all 64 tests match
```

To run the Verilog testbench against your own test vectors with Vivado's
simulator, copy `testbench_template.v`, set `INPUT_WIDTH` / `OUTPUT_WIDTH`, then:

```bash
xvlog circuit.v testbench_template.v && xelab tb_logic_net && xsim tb_logic_net -runall
```

### 4. FPGA synthesis with Vivado

```bash
vivado -mode batch -source examples/synthesis/synthesize.tcl \
    -tclargs circuit.v xc7z020clg400-1 synthesis_reports/
cat synthesis_reports/summary.txt
```

The TCL script accepts four arguments:
```
synthesize.tcl <verilog_file> <part> [output_dir] [clock_period_ns]
```

Common FPGA parts:

| Board | Part |
|-------|------|
| Pynq-Z2 | `xc7z020clg400-1` |
| ZCU104 | `xczu7ev-ffvc1156-2-e` |
| Arty A7-35T | `xc7a35tcpg236-1` |
| Arty A7-100T | `xc7a100tcsg324-1` |

---

## Files

| File | Description |
|------|-------------|
| `synthesize.tcl` | Automated Vivado synthesis script |
| `testbench_template.v` | Verilog testbench template (edit widths before use) |
| `generate_test_vectors.py` | Generate `$readmemb`-format test vectors from a Circuit |

---

## Understanding synthesis results

After synthesis, `synthesis_reports/summary.txt` shows:

- **LUTs** — Look-Up Tables used (primary logic resource)
- **Flip-Flops** — registers; typically 0 for TorchLogix combinational circuits
- **WNS** — Worst Negative Slack; positive = timing met
- **Critical path** — combinational latency end-to-end

TorchLogix circuits contain no registers, so the critical path delay is also
the inference latency.

---

## Troubleshooting

**Testbench output mismatches** — verify that test vectors were generated from
the same circuit version (re-run `generate_test_vectors.py` after any model
change).

**Output X/Z in simulation** — check that all wires in the generated Verilog
are driven; re-run `circuit.simplify()` before `write_verilog_code`.

**Very high LUT count** — expected for large circuits; call `circuit.simplify()`
before export to prune dead gates.

**Timing not met (negative WNS)** — for combinational logic this is
informational; the actual latency is the critical path delay, not the clock
period.
