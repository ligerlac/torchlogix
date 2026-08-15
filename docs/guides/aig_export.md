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
returns an `AIGGraph` — a plain container with the three fields the AIGER
format needs:

```python
aig = circuit.to_and_inverter_graph()

aig.n_inputs   # number of primary inputs
aig.and_gates  # list of (lhs, rhs0, rhs1) literal triples
aig.outputs    # list of output literals, see "Output bit ordering" below
```

Every non-AND gate in the original circuit (`OR`, `XOR`, `NAND`, `WIRE`, …)
is rewritten as one or more two-input ANDs plus literal negation — `XOR`/
`XNOR`, for example, expand to three AND gates each.

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

`aig.outputs` is a flat list of literals, one group per entry in
`circuit.outputs`, in the same order:

- **Plain boolean output** (no `GroupSum`): contributes exactly **one**
  literal — the value of that output bit directly.
- **`GroupSum` score** (a `SumReduction`): contributes **`n_bits`** literals,
  ordered **least-significant bit first**, encoding the integer
  `sum(inputs) + beta` in unsigned binary. `n_bits` is
  `max(1, (len(inputs) + int(beta)).bit_length())`.

For a model with `GroupSum(k=10)`, decode score `j` from its `n_bits`
literals `b[0..n_bits-1]` (LSB first) as:

```python
score_j = sum(bit_value(b[k]) << k for k in range(n_bits))
```

where `bit_value(lit)` looks up the AND-gate/input truth value for
`lit >> 1` and flips it if `lit` is odd (negated), per the AIGER convention
above.

Since every score in a given `GroupSum` layer sums over the same number of
inputs with the same `beta`, they share the same `n_bits` and can be decoded
as consecutive fixed-width fields.

## Limitations

- **`beta` must be a whole number.** `to_and_inverter_graph()` raises
  `ValueError` if any `GroupSum`'s `beta` is fractional — the AIG encoder
  represents `beta` as an integer added into the adder tree.
- **`tau` is not applied.** AND-inverter graphs have no arithmetic for
  non-integer scaling, so the exported bits are the raw
  `sum(inputs) + beta` — they are **not** divided by `tau`. If your model
  uses `GroupSum(tau=...)` with `tau != 1`, apply the division downstream
  after decoding the integer score (e.g. in the code that reads the
  synthesis tool's simulation output).
- **All outputs are unsigned integers or single bits** — there is no
  floating-point output path in the AIG export, unlike `write_c_code()` /
  `write_verilog_code()`, which fall back to a `float` score type when
  `tau != 1` or `beta` is fractional.
