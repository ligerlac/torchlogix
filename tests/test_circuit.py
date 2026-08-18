import pytest
import subprocess
import ctypes
import shutil
import sys
import tempfile
import torch
import torch.nn as nn
from torchlogix import Circuit
from torchlogix.utils import set_export_mode
from torchlogix.layers import (
    GroupSum,
    LogicConv2d,
    LogicConv3d,
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


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel])
def test_functional_equivalence(model_cls):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_model = model(x)
    
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_circuit = circuit(x)
    assert torch.equal(preds_model, preds_circuit.to(preds_model.dtype)), \
        "Circuit predictions differ from Eval-mode model predictions"

@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, AnyLogicModel])
def test_aig_functional_equivalence(model_cls):
    model = model_cls()
    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    with tempfile.NamedTemporaryFile(suffix=".aig") as tmp_file:
        circuit.write_to_aiger_file(tmp_file.name)
        data = open(tmp_file.name, "rb").read()
        nl = data.index(b"\n")
        mode, m, i, l, o, a = data[:nl].decode().split()
        i, l, o, a = int(i), int(l), int(o), int(a)
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
        ands = []
        for gate_idx in range(a):
            var = i + l + gate_idx + 1
            lhs = var * 2
            delta0, pos = read_delta(pos)
            rhs0 = lhs - delta0
            delta1, pos = read_delta(pos)
            rhs1 = rhs0 - delta1
            ands.append((lhs, rhs0, rhs1))



ABC_PATH = shutil.which("abc")


@pytest.mark.skipif(ABC_PATH is None, reason="abc binary not found on PATH")
@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, AnyLogicModel])
def test_third_party_functional_equivalence(model_cls):
    model = model_cls()
    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=model.input_shape)

    with tempfile.NamedTemporaryFile(suffix=".aig") as tmp_file, \
        tempfile.NamedTemporaryFile(suffix=".aig") as tmp_roundtrip:
        circuit.write_to_aiger_file(tmp_file.name)

        result = subprocess.run(
            [ABC_PATH, "-q", f"read_aiger {tmp_file.name}; write_aiger {tmp_roundtrip.name}"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"ABC failed to read and write the circuit: {result.stderr}"

        data = open(tmp_roundtrip.name, "rb").read()
    nl = data.index(b"\n")
    mode, m, i, l, o, a = data[:nl].decode().split()
    i, l, o, a = int(i), int(l), int(o), int(a)
    pos = nl + 1
    outputs = []
    for _ in range(o):
        nl2 = data.index(b"\n", pos)
        outputs.append(int(data[pos:nl2]))
        pos = nl2 + 1


    pass
# create a testing method
# make the circuit, convert to AIG format, then test it?

# assert that the outputs of circuit are identical to the outputs of the AIG graph
# assert that writing to the file also works, take an AIG, write it to a file, read it back from the file
# and assert that the OG one and the one i read from the file are still producing the same outputs
# for this, use mockturtle or ABC

# on meeting on monday, have a short slide/presentation, in bullet points you put the whole idea of the project
# what i have been up to so far, some screenshot of code i wrote, some pictures that shows a simple circuit, what their corresponding AIG graph looks like
# 

@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel])
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


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
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


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
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


