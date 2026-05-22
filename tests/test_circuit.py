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


# inherit from sequential
class ConvModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LogicConv2d(in_dim=32, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2, parametrization_kwargs={"weight_init": "random"}),
            OrPooling2d(kernel_size=2, stride=2),
            nn.Flatten(),  # 8 × 15 x 15 = 1800
            LogicDense(1800, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            GroupSum(10),
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
    
 
@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
@pytest.mark.parametrize("pack_bits", [None, 8, 16, 32])
@pytest.mark.parametrize("relative_batch_size", [1, 10])
def test_circuit_compilation(model_cls, pack_bits, relative_batch_size):
    model = model_cls()

    batch_size = (1 if pack_bits is None else pack_bits) * relative_batch_size
    x = torch.randint(0, 2, (batch_size, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_model = model(x)
    
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_circuit = circuit(x)
    assert torch.equal(preds_model, preds_circuit), "Circuit predictions differ from Eval-mode model predictions"

    circuit.compile(pack_bits=pack_bits)
    input_np = x.numpy()
    preds_circuit_compiled = circuit(input_np, use_compiled=True)
    preds_circuit_compiled_torch = torch.from_numpy(preds_circuit_compiled)
    assert torch.equal(preds_model, preds_circuit_compiled_torch), "Compiled circuit predictions differ from Eval-mode predictions"


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
@pytest.mark.parametrize("simplification", [
    Circuit.simplify, Circuit.constant_fold_gates, Circuit.eliminate_dead_gates, Circuit.bypass_wires, Circuit.dedup
])
def test_circuit_simplifications(model_cls, simplification):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_before = circuit(x.reshape(x.shape[0], -1))

    simplification(circuit)
    preds_after = circuit(x.reshape(x.shape[0], -1))
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
    assert circuit.output_reduction is not None

    r = circuit.output_reduction
    c_code = circuit.get_c_code()

    # GroupSum is now inlined: circuit outputs float scores, no separate circuit_scores.
    assert "float   out[" in c_code
    assert f"for (int j = 0; j < {r.k}; j++)" in c_code
    assert "circuit_scores" not in c_code

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
    assert preds_python.shape[-1] == r.k


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
def test_turn_group_sum_into_argmax(model_cls):
    """turn_group_sum_into_argmax produces one-hot outputs that match Python argmax."""
    model = model_cls()
    x = torch.randint(0, 2, (8, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    circuit.simplify()
    assert circuit.output_reduction is not None

    # Scores from the original circuit.
    scores = circuit(x)
    expected_argmax = scores.argmax(dim=-1)
    expected_one_hot = torch.nn.functional.one_hot(expected_argmax, num_classes=scores.shape[-1]).bool()

    circuit.turn_group_sum_into_argmax()
    circuit.simplify()
    assert circuit.output_reduction is None

    # One-hot from the converted circuit.
    one_hot = circuit(x)  # (8, k) bool

    # one_hot should equal expected_one_hot
    assert torch.equal(one_hot, expected_one_hot), "Circuit one-hot output differs from expected one-hot"

    circuit.compile()
    one_hot_compiled = circuit(x.numpy(), use_compiled=True)

    assert torch.equal(torch.from_numpy(one_hot_compiled), expected_one_hot), "Compiled circuit one-hot output differs from expected one-hot"
