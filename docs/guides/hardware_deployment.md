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

`SCORE_BITS` is 8, 16, 32, or 64 bits. The default keeps the existing raw
Verilog layout and ensures each slot can hold its unscaled count; its width is
independent of the C output type. Score `j` occupies
`scores_flat[j*SCORE_BITS +: SCORE_BITS]`.

By default, Verilog emits **raw group counts**. This preserves the existing
hardware interface even when a model uses non-default `tau` or `beta` during
training. To obtain scaled regression scores in hardware, request the optional
signed fixed-point output mode:

```python
verilog = circuit.get_verilog_code(fixed_point_fractional_bits=16)
# or: circuit.write_verilog_code("circuit.v", fixed_point_fractional_bits=16)
```

In this mode each score occupies 32 bits. Interpret each slice as a signed
two's-complement integer and divide by `2**16` to recover its real value.
The generator uses constant integer multipliers and offsets for each group,
with rounding to the nearest fixed-point step. It rejects scales whose
intermediate or output range would overflow. Choose the fractional bit count
to balance range and precision; 16 bits allow approximately [-32768, 32768)
with steps of `2**-16`. The fixed-point mode is opt-in, so existing raw-count
Verilog consumers keep their current layout and values.

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
