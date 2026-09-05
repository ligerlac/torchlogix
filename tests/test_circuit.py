import itertools
import random
import pytest
import subprocess
import ctypes
import shutil
import sys
import tempfile
import torch
import torch.nn as nn
from torchlogix import Circuit
from torchlogix.circuit import AIGGraph, Gate, GateOp, SumReduction
from torchlogix.utils import set_export_mode
from torchlogix.layers import (
    GroupSum,
    LogicConv2d,
    LogicConv3d,
    LogicConvTranspose3d,
    LogicDense,
    OrPooling2d,
    OrPooling3d,
)


class DenseModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
        )
        self.input_shape = (1000,)


# inherit from sequential
class ConvModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LogicConv2d(in_dim=32, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2, parametrization_kwargs={"weight_init": "random"}),
            OrPooling2d(kernel_size=2, stride=2),
            nn.Flatten(),  # 8 × 15 x 15 = 1800
            LogicDense(1800, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            GroupSum(10)# , tau=2.0),
        )
        self.input_shape = (3, 32, 32)


# w/ custom forward pass
class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=32, channels=3, num_kernels=8,
                    receptive_field_size=3, tree_depth=2,
                    parametrization_kwargs={"weight_init": "random"}) # 8 x 30 x 30 = 7200
        self.pool = OrPooling2d(kernel_size=2, stride=2) # 8 x 15 x 15 = 1800
        self.dense = LogicDense(1801, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"})
        self.group_sum = GroupSum(10)
        self.input_shape = (32*32*3 + 1,)

    def forward(self, x):
        assert x.shape[1:] == (32*32*3 + 1,)
        img, feat = x[:, :-1].reshape(-1, 3, 32, 32), x[:, -1:]
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x


class ConvTransposeAE3dModel(nn.Sequential):
    """3D autoencoder: LogicConv3d halves 6^3 to 3^3, LogicConvTranspose3d restores it.

    Both layers use nonzero padding, and the decoder a nonzero output_padding,
    so this single model covers the whole transposed-conv export path:

    * input dilation (stride > 1) expressed functionally - building a zero
      tensor and writing into it makes constant_fold_views reject the graph
    * padding applied inside FixedConvTransposeConnections rather than by the
      layer, which would otherwise double-pad
    * kernel_positions describing the larger transposed output

    Per-axis shapes and gradients are covered more finely in test_clgn_3d.py;
    this is the end-to-end circuit-level check.
    """

    def __init__(self):
        super().__init__(
            LogicConv3d(in_dim=6, channels=2, num_kernels=3, receptive_field_size=3,
                        tree_depth=2, stride=2, padding=1,
                        parametrization_kwargs={"weight_init": "random"}),      # -> 3 x 3^3
            LogicConvTranspose3d(in_dim=3, channels=3, num_kernels=2, receptive_field_size=3,
                                 tree_depth=2, stride=2, padding=1, output_padding=1,
                                 parametrization_kwargs={"weight_init": "random"}),  # -> 2 x 6^3
            nn.Flatten(),   # Circuit outputs are flat; the spatial round-trip
                            # itself is asserted in test_clgn_3d.py
        )
        self.input_shape = (2, 6, 6, 6)


class InPlaceConstMutationModel(nn.Module):
    """Mutates a constant tensor in place after creation
    (`mask = torch.ones(8, 8); mask[4:, :] = 0`) - torch.fx's constant
    folding can't fold this (see constant_fold_views/_reject_orphaned_impure_ops
    in circuit.py), so from_model must reject it clearly rather than
    silently building a wrong circuit.
    """
    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)

    def forward(self, x):
        mask = torch.ones(8, 8, dtype=x.dtype, device=x.device)
        mask[4:, :] = 0
        return x & mask


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, ConvTransposeAE3dModel])
def test_functional_equivalence(model_cls):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_model = model(x)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_circuit = circuit(x)
    assert torch.equal(preds_model, preds_circuit.to(preds_model.dtype)), \
        "Circuit predictions differ from Eval-mode model predictions"


# ---------------------------------------------------------------------------
# AIG helpers
#
# _eval_aig is an independent, pure-Python AIGER evaluator: given an AIGGraph
# (or anything with .and_gates / .outputs in the same shape) and concrete
# primary-input values, it walks the AND gates and resolves literal polarity
# itself, without going through Circuit at all. This lets tests check AIG
# *function*, not just that a file round-trips through some other tool.
# ---------------------------------------------------------------------------

def _eval_aig(aig: AIGGraph, inputs: list) -> list:
    """Evaluate an AIGGraph in pure Python for one concrete input assignment.

    inputs[k] is the value of AIGER variable k+1 (i.e. Circuit input id k).
    Returns one bool per aig.outputs literal, in order.
    """
    values = {idx + 1: bool(v) for idx, v in enumerate(inputs)}

    def lit_val(lit: int) -> bool:
        if lit == 0:
            return False
        if lit == 1:
            return True
        v = values[lit // 2]
        return (not v) if (lit & 1) else v

    for lhs, rhs0, rhs1 in aig.and_gates:
        values[lhs // 2] = lit_val(rhs0) and lit_val(rhs1)

    return [lit_val(lit) for lit in aig.outputs]


def _decode_aig_outputs(circuit: Circuit, aig_bits: list) -> list:
    """Group a flat list of AIG output bits back into one value per circuit
    output, mirroring the layout Circuit.to_and_inverter_graph() produces:
    a plain boolean output consumes one bit; a SumReduction output consumes
    max(1, (len(input_ids) + int(beta)).bit_length()) bits, LSB first.
    """
    sum_by_id = circuit._sum_by_id
    pos = 0
    decoded = []
    for out_id in circuit.outputs:
        sr = sum_by_id.get(out_id)
        if sr is None:
            decoded.append(int(aig_bits[pos]))
            pos += 1
        else:
            max_value = len(sr.input_ids) + int(round(sr.beta))
            n_bits = max(1, max_value.bit_length())
            bits = aig_bits[pos:pos + n_bits]
            pos += n_bits
            decoded.append(sum((1 << k) for k, bit in enumerate(bits) if bit))
    assert pos == len(aig_bits)
    return decoded


def _decode_via_output_specs(aig: AIGGraph, aig_bits: list) -> list:
    """Same decoding as _decode_aig_outputs, but using only AIGGraph's own
    output_specs -- no access to the source Circuit. This is the point of
    the output ABI: a consumer holding just the AIGGraph can reconstruct
    torchlogix's outputs.
    """
    decoded = []
    for spec in aig.output_specs:
        assert spec.bit_order == "lsb_first"
        bits = aig_bits[spec.start_bit:spec.start_bit + spec.width]
        if spec.kind == "bool":
            decoded.append(int(bits[0]))
        else:
            decoded.append(sum((1 << k) for k, bit in enumerate(bits) if bit))
    return decoded


def _parse_aiger_file(path: str) -> AIGGraph:
    """Parse a binary AIGER (.aig) file back into an AIGGraph. Only supports
    what Circuit.write_to_aiger_file emits: no latches, an ASCII header line,
    one output literal per line, then delta-encoded AND gates.
    """
    data = open(path, "rb").read()
    nl = data.index(b"\n")
    mode, m, i, l, o, a = data[:nl].decode().split()
    assert mode == "aig"
    i, l, o, a = int(i), int(l), int(o), int(a)
    assert l == 0, "latches are not supported"

    pos = nl + 1
    outputs = []
    for _ in range(o):
        nl2 = data.index(b"\n", pos)
        outputs.append(int(data[pos:nl2]))
        pos = nl2 + 1

    def read_delta(pos):
        delta, shift = 0, 0
        while True:
            ch = data[pos]
            pos += 1
            if ch & 0x80:
                delta |= (ch & 0x7F) << shift
            else:
                delta |= ch << shift
                break
            shift += 7
        return delta, pos

    and_gates = []
    for gate_idx in range(a):
        var = i + l + gate_idx + 1
        lhs = var * 2
        delta0, pos = read_delta(pos)
        rhs0 = lhs - delta0
        delta1, pos = read_delta(pos)
        rhs1 = rhs0 - delta1
        and_gates.append((lhs, rhs0, rhs1))

    return AIGGraph(n_inputs=i, and_gates=and_gates, outputs=outputs)


def test_aiger_serializer_rejects_self_referencing_and_gate():
    """AIGGraph.write_to_aiger_file must reject a malformed AND gate with a
    real exception rather than silently emitting a corrupt file.

    The AIGER invariant is the strict lhs > rhs0 >= rhs1. delta0 == 0 (i.e.
    lhs == rhs0, a self-referencing gate) used to pass the old
    `delta0 >= 0` assertion -- and assertions are stripped entirely under
    `python -O`, so this must be a normal exception, not an assert.
    """
    aig = AIGGraph(n_inputs=1, and_gates=[(4, 4, 2)], outputs=[4])  # lhs == rhs0
    with tempfile.NamedTemporaryFile(suffix=".aig") as tmp_file:
        with pytest.raises(ValueError, match="lhs > rhs0 >= rhs1"):
            aig.write_to_aiger_file(tmp_file.name)


def test_aiger_serializer_accepts_valid_and_gate():
    """Sanity check that the stronger validation doesn't reject a
    well-formed AND gate."""
    aig = AIGGraph(n_inputs=2, and_gates=[(6, 4, 2)], outputs=[6])
    with tempfile.NamedTemporaryFile(suffix=".aig") as tmp_file:
        aig.write_to_aiger_file(tmp_file.name)  # must not raise
        assert _parse_aiger_file(tmp_file.name).and_gates == [(6, 4, 2)]


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, ConvTransposeAE3dModel])
def test_aig_functional_equivalence(model_cls):
    """Round-trips a trained model's Circuit through the AIGER file format and
    checks -- via the independent Python AIG evaluator above, not a third-party
    tool -- that decoding the file reproduces circuit()'s output exactly.
    """
    model = model_cls()
    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=model.input_shape)

    with tempfile.NamedTemporaryFile(suffix=".aig") as tmp_file:
        circuit.write_to_aiger_file(tmp_file.name)
        aig = _parse_aiger_file(tmp_file.name)

    assert aig.n_inputs == circuit.n_inputs

    rng = random.Random(0)
    for _ in range(8):
        bits = [rng.random() < 0.5 for _ in range(circuit.n_inputs)]
        x = torch.tensor(bits, dtype=torch.bool).reshape(1, *circuit.input_shape)
        expected = [int(v) for v in circuit(x)[0].tolist()]

        aig_bits = _eval_aig(aig, bits)
        actual = _decode_aig_outputs(circuit, aig_bits)
        assert actual == expected



ABC_PATH = shutil.which("abc")


@pytest.mark.skipif(ABC_PATH is None, reason="abc binary not found on PATH")
@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, ConvTransposeAE3dModel])
def test_abc_reads_and_rewrites_aiger(model_cls):
    """Parser/compatibility check: ABC can read a TorchLogix .aig file and
    write out a functionally equivalent one.

    This intentionally does NOT check the AIG against the source
    circuit/model -- see test_aig_functional_equivalence for that. This test
    only checks that ABC's read/write round trip preserves the AIG's own
    function, i.e. that our AIGER encoding is something a real third-party
    tool can consume without corrupting it.
    """
    model = model_cls()
    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=model.input_shape)

    with tempfile.NamedTemporaryFile(suffix=".aig") as tmp_file, \
        tempfile.NamedTemporaryFile(suffix=".aig") as tmp_roundtrip:
        circuit.write_to_aiger_file(tmp_file.name)
        original = _parse_aiger_file(tmp_file.name)

        result = subprocess.run(
            [ABC_PATH, "-q", f"read_aiger {tmp_file.name}; write_aiger {tmp_roundtrip.name}"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"ABC failed to read and write the circuit: {result.stderr}"

        roundtrip = _parse_aiger_file(tmp_roundtrip.name)

    assert roundtrip.n_inputs == original.n_inputs
    assert len(roundtrip.outputs) == len(original.outputs)

    rng = random.Random(0)
    for _ in range(8):
        bits = [rng.random() < 0.5 for _ in range(original.n_inputs)]
        assert _eval_aig(roundtrip, bits) == _eval_aig(original, bits), \
            "ABC's read/write round trip changed the AIG's function"


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, ConvTransposeAE3dModel])
@pytest.mark.parametrize("pack_bits", [None, 8, 16, 32])
@pytest.mark.parametrize("relative_batch_size", [1, 10])
def test_circuit_compilation(model_cls, pack_bits, relative_batch_size):
    model = model_cls()

    batch_size = (1 if pack_bits is None else pack_bits) * relative_batch_size
    x = torch.randint(0, 2, (batch_size, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_model = model(x)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    circuit.compile(pack_bits=pack_bits)
    input_np = x.numpy()
    preds_circuit_compiled = circuit(input_np, use_compiled=True)
    preds_circuit_compiled_torch = torch.from_numpy(preds_circuit_compiled)
    # Cast to a common dtype before comparing: circuit may use a narrower integer
    # type (e.g. uint16_t) while the model returns float32.
    target_dtype = preds_model.dtype
    assert torch.equal(preds_model, preds_circuit_compiled_torch.to(target_dtype)), \
        "Compiled circuit predictions differ from Eval-mode predictions"


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel, ConvTransposeAE3dModel])
@pytest.mark.parametrize("simplification", [
    Circuit.simplify, Circuit.constant_fold_gates, Circuit.eliminate_dead_gates, Circuit.bypass_wires, Circuit.dedup, Circuit.fuse_not_inputs
])
def test_circuit_simplifications(model_cls, simplification):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_before = circuit(x)

    simplification(circuit)
    preds_after = circuit(x)
    assert torch.equal(preds_before, preds_after), f"Predictions differ after {simplification.__name__}!"


def test_rejects_inplace_constant_mutation():
    model = InPlaceConstMutationModel()
    with pytest.raises(NotImplementedError, match="unsupported constant-tensor mutation"):
        Circuit.from_model(model, input_shape=model.input_shape)


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel, ConvTransposeAE3dModel])
def test_json_roundtrip(model_cls):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_before = circuit(x.reshape(x.shape[0], -1))

    # Export the circuit to a temporary file and load it back
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
        circuit.write_json(tmp_file.name)
        circuit_loaded = Circuit.from_json_file(tmp_file.name)

    preds_after = circuit_loaded(x.reshape(x.shape[0], -1))
    assert torch.equal(preds_before, preds_after), "Predictions differ after export/import roundtrip!"



@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
def test_c_codegen_group_sum_scores(model_cls):
    """GroupSum reduction is inlined into circuit and compiles cleanly."""
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    assert circuit.sum_nodes

    from torchlogix.circuit import _c_output_dtype
    sum_by_id = circuit._sum_by_id
    red_outs = [sum_by_id[oid] for oid in circuit.outputs if oid in sum_by_id]
    k = len(red_outs)
    out_dtype = _c_output_dtype(red_outs)
    c_code = circuit.get_c_code()

    assert f"{out_dtype}   out[" in c_code
    assert "bool raw[" in c_code
    assert c_code.count("// --- outputs ---") == 1
    assert c_code.count("int s = 0;") == k

    # Verify it compiles cleanly.
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as tf:
        tf.write(c_code)
        c_path = tf.name
    result = subprocess.run(
        ["gcc", "-std=c99", "-fsyntax-only", c_path],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"C compile error:\n{result.stderr}"

    # Verify scores match Python circuit.
    preds_python = circuit(x.reshape(1, -1))  # shape (1, k)
    assert preds_python.shape[-1] == k


# ---------------------------------------------------------------------------
# Small, hand-built circuits: exhaustive truth-table coverage.
#
# The tests above exercise Circuit end-to-end through real trained models with
# random inputs. The tests below instead build tiny Circuit objects by hand
# (bypassing from_model/tracing entirely) so every input combination can be
# enumerated exactly, for the cases that matter most to get right: every
# GateOp, constant/passthrough outputs, simplify()'s effect on function,
# SumReduction, mixed boolean+reduction outputs, and tau/beta edge cases.
# ---------------------------------------------------------------------------

GATE_ORACLE = {
    GateOp.CONST_FALSE: lambda a, b: False,
    GateOp.CONST_TRUE:  lambda a, b: True,
    GateOp.WIRE:        lambda a, b: a,
    GateOp.NOT:         lambda a, b: not a,
    GateOp.NOT_A:       lambda a, b: not a,
    GateOp.NOT_B:       lambda a, b: not b,
    GateOp.AND:         lambda a, b: a and b,
    GateOp.OR:          lambda a, b: a or b,
    GateOp.XOR:         lambda a, b: a != b,
    GateOp.NAND:        lambda a, b: not (a and b),
    GateOp.NOR:         lambda a, b: not (a or b),
    GateOp.XNOR:        lambda a, b: a == b,
    GateOp.AND_NOT_B:   lambda a, b: a and not b,
    GateOp.AND_NOT_A:   lambda a, b: (not a) and b,
    GateOp.OR_NOT_B:    lambda a, b: a or not b,
    GateOp.OR_NOT_A:    lambda a, b: (not a) or b,
}


def test_gate_oracle_covers_all_gate_ops():
    """Guards the table above itself: if GateOp ever grows a new member,
    this fails until GATE_ORACLE (and therefore test_gate_op_truth_tables)
    is updated to cover it.
    """
    assert set(GATE_ORACLE) == set(GateOp)


def _single_gate_circuit(op: GateOp) -> Circuit:
    """A minimal 2-input, 1-gate circuit: out = op(in0, in1)."""
    circuit = Circuit(n_inputs=2, input_shape=[2])
    circuit.gates = [Gate(gate_id=2, op=op, in0=0, in1=1)]
    circuit.outputs = [2]
    circuit.output_shape = [1]
    return circuit


@pytest.mark.parametrize("op", list(GateOp))
def test_gate_op_truth_tables(op):
    """Exhaustively checks every GateOp's truth table (all 4 input
    combinations) against an independent oracle, both via Circuit's Python
    evaluator and via its AIG (AND-inverter) lowering.
    """
    circuit = _single_gate_circuit(op)
    aig = circuit.to_and_inverter_graph()

    for a, b in itertools.product([False, True], repeat=2):
        expected = GATE_ORACLE[op](a, b)

        x = torch.tensor([[a, b]], dtype=torch.bool)
        actual = bool(circuit(x)[0, 0])
        assert actual == expected, f"{op}: circuit({a}, {b}) = {actual}, expected {expected}"

        aig_actual = _eval_aig(aig, [a, b])[0]
        assert aig_actual == expected, f"{op}: AIG({a}, {b}) = {aig_actual}, expected {expected}"


def test_constant_and_direct_input_outputs():
    """Outputs that are direct passthroughs of a primary input (no gates at
    all), and outputs that are CONST_TRUE/CONST_FALSE gates (independent of
    every input).
    """
    # Direct passthrough: outputs reference input ids with zero gates.
    circuit = Circuit(n_inputs=2, input_shape=[2])
    circuit.outputs = [0, 1]
    circuit.output_shape = [2]
    aig = circuit.to_and_inverter_graph()

    for a, b in itertools.product([False, True], repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.bool)
        assert circuit(x)[0].tolist() == [a, b]
        assert _eval_aig(aig, [a, b]) == [a, b]

    # Constant outputs, independent of the (unused) input.
    circuit = Circuit(n_inputs=1, input_shape=[1])
    circuit.gates = [
        Gate(gate_id=1, op=GateOp.CONST_TRUE),
        Gate(gate_id=2, op=GateOp.CONST_FALSE),
    ]
    circuit.outputs = [1, 2]
    circuit.output_shape = [2]
    aig = circuit.to_and_inverter_graph()

    for a in (False, True):
        x = torch.tensor([[a]], dtype=torch.bool)
        assert circuit(x)[0].tolist() == [True, False]
        assert _eval_aig(aig, [a]) == [True, False]


def _redundant_gate_circuit() -> Circuit:
    """out0 = x AND y, out1 = a structurally different but functionally
    identical expression (double-NOT and a WIRE), so simplify() has real
    dead code and duplicate structure to collapse.
    """
    circuit = Circuit(n_inputs=2, input_shape=[2])
    circuit.gates = [
        Gate(gate_id=2, op=GateOp.NOT, in0=1),          # not y
        Gate(gate_id=3, op=GateOp.NOT, in0=2),           # not (not y) == y
        Gate(gate_id=4, op=GateOp.AND, in0=0, in1=3),    # x and y
        Gate(gate_id=5, op=GateOp.WIRE, in0=0),          # wire x
        Gate(gate_id=6, op=GateOp.WIRE, in0=3),          # wire y
        Gate(gate_id=7, op=GateOp.AND, in0=5, in1=6),    # x and y, again
    ]
    circuit.outputs = [4, 7]
    circuit.output_shape = [2]
    return circuit


def test_circuit_simplify_preserves_truth_table():
    circuit = _redundant_gate_circuit()
    gates_before = len(circuit.gates)

    truth_before = {}
    for a, b in itertools.product([False, True], repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.bool)
        truth_before[(a, b)] = circuit(x)[0].tolist()

    circuit.simplify()
    assert len(circuit.gates) < gates_before, "simplify() should have removed redundant gates"

    for a, b in itertools.product([False, True], repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.bool)
        assert circuit(x)[0].tolist() == truth_before[(a, b)], \
            f"simplify() changed the truth table at ({a}, {b})"


def test_sum_reduction_truth_table():
    circuit = Circuit(n_inputs=3, input_shape=[3])
    circuit.sum_nodes = [SumReduction(node_id=3, input_ids=[0, 1, 2])]
    circuit.outputs = [3]
    circuit.output_shape = [1]
    aig = circuit.to_and_inverter_graph()

    for bits in itertools.product([False, True], repeat=3):
        x = torch.tensor([list(bits)], dtype=torch.bool)
        expected = sum(bits)
        assert int(circuit(x)[0, 0]) == expected

        aig_bits = _eval_aig(aig, list(bits))
        assert _decode_aig_outputs(circuit, aig_bits) == [expected]


def test_aig_output_specs_describe_bit_layout():
    """AIGGraph.output_specs is the output ABI requested in review: for each
    logical Circuit output it must retain which AIG bits belong to it, its
    width, whether it's boolean or numeric, its bit order, and its tau/beta
    -- so a consumer holding only the AIGGraph (not the source Circuit) can
    reconstruct torchlogix's outputs. See docs/guides/aig_export.md.
    """
    circuit = Circuit(n_inputs=3, input_shape=[3])
    circuit.gates = [Gate(gate_id=3, op=GateOp.AND, in0=0, in1=1)]
    circuit.sum_nodes = [SumReduction(node_id=4, input_ids=[0, 1, 2])]
    circuit.outputs = [3, 4]
    circuit.output_shape = [2]
    aig = circuit.to_and_inverter_graph()

    assert aig.output_shape == [2]
    assert len(aig.output_specs) == 2

    bool_spec = aig.output_specs[0]
    assert (bool_spec.start_bit, bool_spec.width, bool_spec.kind) == (0, 1, "bool")
    assert bool_spec.bit_order == "lsb_first"
    assert (bool_spec.tau, bool_spec.beta) == (1.0, 0.0)

    sum_spec = aig.output_specs[1]
    # 3 inputs, all boolean -> max value 3 -> 2 bits, immediately after the
    # 1-bit boolean output above.
    assert (sum_spec.start_bit, sum_spec.width, sum_spec.kind) == (1, 2, "uint")
    assert (sum_spec.tau, sum_spec.beta) == (1.0, 0.0)

    for a, b, c in itertools.product([False, True], repeat=3):
        x = torch.tensor([[a, b, c]], dtype=torch.bool)
        expected = circuit(x)[0].tolist()

        aig_bits = _eval_aig(aig, [a, b, c])
        assert _decode_via_output_specs(aig, aig_bits) == expected


def test_mixed_boolean_and_reduction_outputs():
    """A single circuit whose outputs mix a plain boolean gate and a
    SumReduction, in the same order documented in docs/guides/aig_export.md.
    """
    circuit = Circuit(n_inputs=3, input_shape=[3])
    circuit.gates = [Gate(gate_id=3, op=GateOp.AND, in0=0, in1=1)]
    circuit.sum_nodes = [SumReduction(node_id=4, input_ids=[0, 1, 2])]
    circuit.outputs = [3, 4]  # boolean output, then a reduction output
    circuit.output_shape = [2]
    aig = circuit.to_and_inverter_graph()

    for a, b, c in itertools.product([False, True], repeat=3):
        x = torch.tensor([[a, b, c]], dtype=torch.bool)
        expected = [int(a and b), int(a) + int(b) + int(c)]
        assert circuit(x)[0].tolist() == expected

        aig_bits = _eval_aig(aig, [a, b, c])
        assert _decode_aig_outputs(circuit, aig_bits) == expected


def _random_small_circuit(seed: int, n_inputs: int = 4, n_gates: int = 15) -> Circuit:
    rng = random.Random(seed)
    circuit = Circuit(n_inputs=n_inputs, input_shape=[n_inputs])
    next_id = n_inputs
    for _ in range(n_gates):
        op = rng.choice(list(GateOp))
        in0 = rng.randrange(next_id)
        in1 = rng.randrange(next_id)
        circuit.gates.append(Gate(gate_id=next_id, op=op, in0=in0, in1=in1))
        next_id += 1
    circuit.outputs = sorted(rng.sample(range(n_inputs, next_id), 3))
    circuit.output_shape = [len(circuit.outputs)]
    return circuit


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_small_circuit_aig_equivalence(seed):
    """A handful of small, randomly-wired circuits (all 16 GateOps, arbitrary
    fan-in DAGs), exhaustively checked against their AIG lowering.
    """
    circuit = _random_small_circuit(seed)
    aig = circuit.to_and_inverter_graph()

    for bits in itertools.product([False, True], repeat=circuit.n_inputs):
        x = torch.tensor([list(bits)], dtype=torch.bool)
        expected = [int(v) for v in circuit(x)[0].tolist()]

        aig_bits = _eval_aig(aig, list(bits))
        assert _decode_aig_outputs(circuit, aig_bits) == expected


@pytest.mark.parametrize("beta", [0.5, 1.0, -1.0])
def test_aig_export_rejects_nonzero_beta(beta):
    """Unsupported case: to_and_inverter_graph() requires beta == 0 exactly,
    not merely a whole number -- AIG export does not represent any offset,
    whole or fractional. See docs/guides/aig_export.md.
    """
    circuit = Circuit(n_inputs=2, input_shape=[2])
    circuit.sum_nodes = [SumReduction(node_id=2, input_ids=[0, 1], beta=beta)]
    circuit.outputs = [2]
    circuit.output_shape = [1]

    with pytest.raises(ValueError, match="tau == 1 and beta == 0"):
        circuit.to_and_inverter_graph()


def test_aig_export_rejects_nonzero_tau():
    """Unsupported case: to_and_inverter_graph() requires tau == 1 exactly.
    An AND-inverter graph cannot represent the division by tau, so export
    must fail loudly rather than silently return an unscaled integer sum
    that doesn't match circuit()'s real output.
    """
    circuit = Circuit(n_inputs=2, input_shape=[2])
    circuit.sum_nodes = [SumReduction(node_id=2, input_ids=[0, 1], tau=2.0)]
    circuit.outputs = [2]
    circuit.output_shape = [1]

    with pytest.raises(ValueError, match="tau == 1 and beta == 0"):
        circuit.to_and_inverter_graph()


def test_aig_export_accepts_default_tau_beta():
    """Supported case: the only tau/beta AIG export handles is the default
    tau == 1, beta == 0, in which case the AIG encodes the raw integer sum
    of the reduction's inputs.
    """
    circuit = Circuit(n_inputs=2, input_shape=[2])
    circuit.sum_nodes = [SumReduction(node_id=2, input_ids=[0, 1])]
    circuit.outputs = [2]
    circuit.output_shape = [1]

    aig = circuit.to_and_inverter_graph()  # must not raise
    for bits in itertools.product([False, True], repeat=2):
        aig_bits = _eval_aig(aig, list(bits))
        assert _decode_aig_outputs(circuit, aig_bits) == [sum(bits)]
