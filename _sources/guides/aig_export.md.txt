# AIG Export Guide

TorchLogix can export a trained model as a binary **AIGER** file (`.aig`), the
standard And-Inverter Graph format read by logic-synthesis and verification
tools such as [ABC](https://github.com/berkeley-abc/abc) and
[mockturtle](https://github.com/lsils/mockturtle). This guide walks through
the export pipeline end to end, without modifying any TorchLogix source code.

---

## 1. Build a `Circuit`

`Circuit.from_model` traces a model that has been put into export mode and
unrolls it into a flat gate list:

```python
from torchlogix import Circuit
from torchlogix.utils import set_export_mode

set_export_mode(model)                                    # required before tracing
circuit = Circuit.from_model(model, input_shape=(1, 28, 28))
```

`Circuit` requires **binary inputs**; binarize at the dataset level before
tracing, since binarization layers are not exported.

## 2. Simplify (optional)

`circuit.simplify()` constant-folds, dedups, and removes dead gates. It has
no effect on the circuit's function, only its size, so it's safe (and
recommended) to run before export — fewer gates means a smaller `.aig` file
and less work for downstream synthesis tools:

```python
circuit.simplify()
```

## 3. Convert to an `AIGGraph`

`circuit.to_and_inverter_graph()` lowers every gate to AND/inverter form and
returns an `AIGGraph`:

```python
aig = circuit.to_and_inverter_graph()

aig.n_inputs      # number of primary inputs
aig.and_gates     # list of (lhs, rhs0, rhs1) literal triples
aig.outputs       # flat list of output literals — the AIGER-format payload
aig.output_specs  # list[AIGOutputSpec] — how to regroup aig.outputs into
                   # circuit.outputs, see "Output bit ordering" below
aig.output_shape  # copy of circuit.output_shape
```

Every non-AND gate in the original circuit (`OR`, `XOR`, `NAND`, `WIRE`, …)
is rewritten as one or more two-input ANDs plus literal negation — `XOR`/
`XNOR`, for example, expand to three AND gates each.

`aig.outputs`, `aig.and_gates`, and the header counts (`M I L O A`) are what
gets written to the `.aig` file — that's the whole AIGER format, and it's
what ABC/mockturtle read. `aig.output_specs` and `aig.output_shape` are
**not** written to the file; they only exist on the Python `AIGGraph` object.
See "Output bit ordering" below for what that means for a consumer that only
has the `.aig` file.

## 4. Write the `.aig` file

```python
aig.write_to_aiger_file("circuit.aig")
```

Or skip step 3 and call the shortcut on `Circuit` directly, which does both
steps for you:

```python
circuit.write_to_aiger_file("circuit.aig")
```

The file is written in the binary AIGER format (header line `aig M I L O A`
followed by delta-encoded AND gates). TorchLogix circuits are purely
combinational, so the latch count `L` is always `0`.

## 5. Read the file with ABC or mockturtle

**ABC:**

```bash
abc -q "read_aiger circuit.aig; print_stats"
```

**mockturtle** (C++):

```cpp
#include <mockturtle/mockturtle.hpp>

mockturtle::aig_network aig;
lorina::read_aiger("circuit.aig", mockturtle::aiger_reader(aig));
```

Both tools follow the AIGER literal convention: variable `v`'s **positive**
literal is `2*v` and its **negative** (inverted) literal is `2*v + 1`;
variable `0` is reserved so literal `0` means constant `False` and literal
`1` means constant `True`. Primary input `i` (0-indexed, as passed to
`Circuit.from_model`) is AIGER variable `i + 1`.

## 6. Output bit ordering and decoding

`aig.outputs` is a flat list of literals with no embedded structure — a
10-class `GroupSum` model turns into 70+ anonymous output wires in there.
`aig.output_specs` is the **output ABI**: one `AIGOutputSpec` per entry in
`circuit.outputs`, in the same order, telling you exactly how to regroup
`aig.outputs` back into the original values:

```python
@dataclass
class AIGOutputSpec:
    start_bit: int    # aig.outputs[start_bit : start_bit + width] is this output
    width:     int    # 1 for a boolean output
    kind:      str    # "bool" or "uint"
    bit_order: str = "lsb_first"
    tau:       float = 1.0   # always 1.0 -- see Limitations
    beta:      float = 0.0   # always 0.0 -- see Limitations
```

- **`kind="bool"`** (plain boolean output, no `GroupSum`): `width` is always
  `1` — the value of that output bit directly.
- **`kind="uint"`** (a `GroupSum` score / `SumReduction`): `width` is
  `max(1, len(inputs).bit_length())` bits, ordered **least-significant bit
  first**, encoding the unsigned integer `sum(inputs)`.

Decoding with `output_specs` in Python:

```python
def decode(aig, aig_output_bits):
    values = []
    for spec in aig.output_specs:
        bits = aig_output_bits[spec.start_bit : spec.start_bit + spec.width]
        if spec.kind == "bool":
            values.append(bits[0])
        else:  # "uint", lsb_first
            values.append(sum(b << k for k, b in enumerate(bits)))
    return values
```

where `aig_output_bits[k]` is the resolved boolean value of `aig.outputs[k]`
(look up the AND-gate/input truth value for `lit >> 1` and flip it if `lit`
is odd — per the AIGER literal convention above).

**Important:** `output_specs`/`output_shape` live only on the in-memory
`AIGGraph` — they are *not* written into the `.aig` file (which is plain
AIGER, readable by any AIGER-compliant tool, with no room for this
metadata). A consumer that reads the `.aig` file directly (e.g. from ABC or
mockturtle, without going through TorchLogix) does not get this layout for
free; it must be told out of band, e.g. by keeping the `AIGGraph`/`Circuit`
Python object around, or agreeing on the layout ahead of time from
`circuit.output_shape` and the `GroupSum` sizes in the model. Persisting
`output_specs` into the file itself (e.g. via AIGER output symbols, or a
JSON sidecar) is a natural extension but is not implemented yet.

## Limitations

- **AIG export requires `tau == 1` and `beta == 0` exactly.**
  `to_and_inverter_graph()` raises `ValueError` for any `GroupSum` output
  where that doesn't hold. Earlier versions tolerated any whole-number
  `beta` and silently ignored `tau != 1` (encoding a raw sum that didn't
  match the model's real output) — that was a trap, not a feature, so it's
  now a hard error instead. This is a deliberate scope limit, not a bug: an
  And-Inverter Graph has no arithmetic for scaling or offsetting a value, so
  there is no way for AIG export to represent `(sum + beta) / tau` inside
  the graph itself, unlike `write_c_code()` / `write_verilog_code()`, whose
  output formats *can* express that arithmetic. If your model needs
  non-default `tau`/`beta`, use the C or Verilog export path instead of AIG.
- **All outputs are unsigned integers or single bits** — there is no
  floating-point output path in the AIG export, unlike `write_c_code()` /
  `write_verilog_code()`, which fall back to a `float` score type when
  `tau != 1` or `beta` is fractional.
