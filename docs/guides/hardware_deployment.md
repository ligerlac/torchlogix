# Hardware Deployment Guide

TorchLogix can generate synthesizable Verilog RTL directly from any trained
model via `circuit.get_verilog_code()`. This guide explains the generated
interface and how to take it to simulation or an FPGA.

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

## Simulation with Verilator

`examples/verify_with_verilator.py` builds a small circuit, generates Verilog,
and verifies that Verilator simulation matches Python output exactly:

```bash
# Install verilator first
# macOS:   brew install verilator
# Ubuntu:  sudo apt install verilator

python examples/verify_with_verilator.py
# → PASS: all 64 tests match
```

---

## FPGA synthesis with Vivado

Write the Verilog to a file and run the TCL script in `examples/synthesis/`:

```python
circuit.write_verilog_code("circuit.v")
```

```bash
vivado -mode batch -source examples/synthesis/synthesize.tcl \
       -tclargs circuit.v xc7z020clg400-1 results/
```

See [`examples/synthesis/README.md`](../../examples/synthesis/README.md) for the
full workflow: test-vector generation, choosing an FPGA part, and interpreting
synthesis reports.

---

## Design considerations

| Property | Value |
|----------|-------|
| Combinational depth | proportional to network depth (one gate per LUT tree level) |
| Critical path | dominated by the deepest gate chain; use `circuit.simplify()` to reduce gate count before export |
| GroupSum | synthesizes as an integer adder tree; synthesis tools map efficiently to carry chains |
| Timing | no registers in generated RTL; add pipeline registers in post-processing if needed |
