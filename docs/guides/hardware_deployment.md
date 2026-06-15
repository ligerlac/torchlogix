# Hardware Deployment Guide

TorchLogix can generate synthesizable Verilog RTL directly from any trained
model via `circuit.get_verilog_code()`. This guide explains the generated
interface, how to simulate it with Verilator, and how to target an FPGA.

---

## Verilog interface

`circuit.get_verilog_code()` returns a combinational `module circuit` with:

**Boolean-only models** (no `GroupSum`):

```verilog
module circuit (
    input  wire [N_IN-1:0]  inp,
    output wire [N_OUT-1:0] out
);
```

**Models with `GroupSum`** (score outputs):

```verilog
// scores_flat = N_CLASSES × SCORE_BITS packed integer bus
module circuit (
    input  wire [N_IN-1:0]               inp,
    output reg  [N_CLASSES*SCORE_BITS-1:0] scores_flat
);
```

`SCORE_BITS` is the narrowest unsigned integer type that fits the maximum
possible sum (8, 16, 32, or 64 bits, or 32-bit float when `tau ≠ 1`). Score `j`
occupies `scores_flat[j*SCORE_BITS +: SCORE_BITS]`.

---

## Simulating with Verilator

The script `examples/verify_with_verilator.py` builds a small circuit, generates
Verilog, and verifies that Verilator simulation matches Python output exactly.
Run it directly:

```bash
# Install verilator first
# macOS:   brew install verilator
# Ubuntu:  sudo apt install verilator

python examples/verify_with_verilator.py
# → PASS: all 64 tests match
```

### How it works

```python
from torchlogix import Circuit
from torchlogix.utils import set_export_mode

set_export_mode(model)
circuit = Circuit.from_model(model, input_shape=(8,))
circuit.simplify()

# 1 — generate Verilog
verilog = circuit.get_verilog_code()

# 2 — run Python circuit on test inputs
py_out = circuit(x_torch).numpy()   # reference outputs

# 3 — build & run Verilator simulation (see examples/verify_with_verilator.py)
# ...
# PASS: all 64 tests match
```

The Verilator C++ testbench drives `inp`, calls `eval()`, and compares `out`
against the Python reference. For circuits with `scores_flat`, read each
`SCORE_BITS`-wide slice and compare to the Python integer scores.

---

## FPGA synthesis

Write the Verilog to a file and synthesize with any standard RTL flow:

```python
circuit.write_verilog_code("circuit.v")
```

### Vivado (Xilinx / AMD)

Use the TCL script in `examples/synthesis/synthesize.tcl`:

```bash
vivado -mode batch -source examples/synthesis/synthesize.tcl \
       -tclargs circuit.v xc7z020clg400-1 results/
```

See `examples/synthesis/README.md` for a complete walkthrough including
test-vector generation and timing interpretation.

### Yosys / nextpnr (open source)

```bash
yosys -p "read_verilog circuit.v; synth -top circuit; write_json circuit.json"
nextpnr-ice40 --hx8k --json circuit.json --asc circuit.asc
```

---

## Generating test vectors

`examples/synthesis/generate_test_vectors.py` creates binary test data for
Verilog testbenches:

```bash
python examples/synthesis/generate_test_vectors.py \
    --model trained_model.pt --input-shape 1 28 28 \
    --num-tests 1000 --seed 42
```

This writes `test_inputs.txt` and `test_outputs.txt` in `$readmemb` format,
ready to use in a Verilog testbench.

---

## Design considerations

| Property | Value |
|----------|-------|
| Combinational depth | proportional to network depth (one gate per LUT tree level) |
| Critical path | dominated by the deepest gate chain; use `circuit.simplify()` to reduce gate count before export |
| GroupSum | synthesizes as an integer adder tree; synthesis tools map efficiently to carry chains |
| Timing | no registers in generated RTL; add pipeline registers in post-processing if needed |
