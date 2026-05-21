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
def test_circuit_compilation(model_cls, pack_bits):
    model = model_cls()

    batch_size = 1 if pack_bits is None else pack_bits
    x = torch.randint(0, 2, (batch_size, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_eager = model(x)
    
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_circuit = circuit(x.reshape(x.shape[0], -1))
    assert torch.equal(preds_eager, preds_circuit), "Circuit predictions differ from Eval-mode predictions"

    circuit.compile(pack_bits=pack_bits)
    input_np = x.reshape(x.shape[0], -1).numpy()
    preds_circuit_compiled = circuit(input_np, use_compiled=True)
    preds_circuit_compiled_torch = torch.from_numpy(preds_circuit_compiled)
    assert torch.equal(preds_eager, preds_circuit_compiled_torch), "Compiled circuit predictions differ from Eval-mode predictions"


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
