import pytest
import subprocess
import ctypes
import sys
import tempfile
import torch
import torch.nn as nn
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
    

# @pytest.mark.parametrize("model_class", [DenseModel, ConvModel, BranchModel])
@pytest.mark.parametrize("model_class", [DenseModel])
def test_round_trip(model_class):
    # write model to disk, reaload it and assert that the outputs are the same
    x = torch.randn(1, *model_class().input_shape)

    model = model_class()
    out_original = model(x)

    # safe to file
    model_loaded = model_class()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_loaded.load_state_dict(torch.load(tmp.name))

    out_loaded = model_loaded(x)
    assert torch.allclose(out_original, out_loaded, atol=1e-5), "Original and loaded outputs do not match"
