import pytest
import subprocess
import ctypes
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


class AnyLogicModel(nn.Module):
    """
    Some random non-torchlogix logic and reshaping operations
    to test the flexibility of from_model
    """
    def __init__(self):
        super().__init__()
        self.input_shape = (4, 8, 8)


    def forward(self, x):
        # x: (B, 4, 8, 8) — batch of 4-channel 8×8 bool/int tensors

        # Split along channel dim
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]  # each (B, 8, 8)

        # Spatial flip on x1 (replaces the buggy ::-1 step)
        x1 = torch.flip(x1, dims=[1])                           # flip rows → (B, 8, 8)

        # Permute x2: swap H and W
        x2 = x2.permute(0, 2, 1)                                # (B, 8, 8) transposed

        # Broadcast a mask over x3 — zero out the bottom half
        mask = torch.ones(8, 8, dtype=x3.dtype, device=x3.device)
        mask[4:, :] = 0                                         # (8, 8), broadcasts over batch
        x3 = x3 & mask

        # Diagonal mask on x4 — keep only upper triangle
        tri = torch.triu(torch.ones(8, 8, dtype=x4.dtype, device=x4.device))
        x4 = x4 & tri

        # Logic ops: operator precedence is & > ^ > |  (same as Python/C)
        # So this reads as:  x1 | (x2 & x3) ^ x4
        # Use parens to make intent explicit:
        out = x1 | ((x2 & x3) ^ x4)                            # (B, 8, 8)

        # Add a channel dim back, then roll it to position 1
        out = out.unsqueeze(1)                                   # (B, 1, 8, 8)

        # Flatten spatial dims only
        out = out.flatten(2)                                     # (B, 1, 64)

        out = out.squeeze(1)                                     # (B, 64)

        out1 = out[:, :8].sum(dim=1, keepdim=True)               # (B, 1)
        out2 = out[:, 8:16].sum(dim=1, keepdim=True)             # (B, 1)
        out3 = out[:, 16:]                                       # (B, 16)

        return torch.cat([out1, out2, out3], dim=1)                      # (B, 18)


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, AnyLogicModel])
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


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel, AnyLogicModel])
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


