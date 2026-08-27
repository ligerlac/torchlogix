# Hardware Deployment Guide

TorchLogix supports two paths from a trained model to hardware:

- **`Circuit`** (native, this guide) - a minimal, logic-only IR: always
  fully unrolled (no control flow, no registers), entirely generated and
  controlled by `torchlogix` itself. Good default for straightforward
  combinational export to C or Verilog.
- **[`alkaid`](#compiling-with-alkaid)** (optional integration) - a
  general-purpose compiler for low-latency static-dataflow FPGA kernels
  (RTL, HLS, XLS targets). More advanced and more configurable, at the cost
  of an extra dependency and a less direct code path.

---

# `Circuit` (native)

TorchLogix can generate synthesizable Verilog RTL directly from any trained
model via `circuit.get_verilog_code()`. This section explains the generated
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

---

For synthesis/verification flows built around And-Inverter Graphs instead of
RTL (e.g. ABC, mockturtle), see the [AIG Export Guide](aig_export.md).

## Compiling with `alkaid`

[`alkaid`](https://github.com/calad0i/alkaid) is a 3rd party compiler for
generating low-latency static-dataflow kernels for FPGAs (RTL, HLS, XLS
targets). Unlike `Circuit`, it's a general-purpose tool not specific to
`torchlogix` or even to boolean logic - `torchlogix` integrates with it as
an optional extra for more advanced FPGA compilation flows.

### Install

```bash
pip install torchlogix[alkaid]
```

### Usage

```python
from alkaid.converter import trace_model
from alkaid.trace import FVArrayInput, trace

from torchlogix.utils import set_export_mode

set_export_mode(model)  # model is any torchlogix nn.Module, in eval mode

inp = FVArrayInput((1, *model.input_shape)).quantize(0, 1, 0)
inp2, out = trace_model(model, inputs=inp, framework="logic")
comb = trace(inp2, out)

comb.predict(x.numpy())  # x: a torch.bool tensor matching model.input_shape
```

`framework="logic"` selects `torchlogix`'s tracer, registered with `alkaid`
via its `alir_tracer.plugins` entry point.

### How the plugin works

The tracer itself (`torchlogix._alkaid_plugin`) has no dependency on
`torchlogix` beyond that entry-point registration. It traces the model's
export-mode `forward()` with `torch.fx`'s `make_fx` at the aten-op level, then
replays the resulting graph on `alkaid`'s `FVArray`. Since it only ever sees
generic aten ops (not torchlogix layers), it works for any PyTorch model built
from pure boolean/integer operations - not just `torchlogix` models.
