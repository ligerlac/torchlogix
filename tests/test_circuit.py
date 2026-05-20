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
def test_circuit(model_cls):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    model.eval()
    preds_eval = model(x)
    set_export_mode(model, enabled=True, batch_size=1)
    preds_export = model(x)
    assert torch.equal(preds_eval, preds_export), "Export-mode predictions differ from Eval-mode predictions"
    
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_circuit = circuit(x.reshape(x.shape[0], -1))
    assert torch.equal(preds_eval, preds_circuit), "Circuit predictions differ from Eval-mode predictions"

    circuit.compile()
    preds_circuit_compiled = circuit(x.reshape(x.shape[0], -1))
    assert torch.equal(preds_eval, preds_circuit_compiled), "Compiled circuit predictions differ from Eval-mode predictions"
