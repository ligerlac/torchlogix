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
    LearnableBinarization
)


class DenseModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LearnableBinarization(thresholds=[0.33, 0.66]),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
        )
        self.input_shape = (500,)


# inherit from sequential
class ConvModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LearnableBinarization(thresholds=[0.33, 0.66], one_per="channel", feature_dim=1),
            LogicConv2d(in_dim=32, channels=6, num_kernels=8, receptive_field_size=3, tree_depth=2, parametrization_kwargs={"weight_init": "random"}),
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
    

@pytest.mark.parametrize("model_class", [DenseModel, ConvModel, BranchModel])
def test_eval_mode_equivalence(model_class):
    x = torch.randn(1, *model_class().input_shape)

    model = model_class()

    # set params to extreme values such that relaxed and discrete version should be the same
    for module in model.modules():
        if isinstance(module, LearnableBinarization):
            # low temperatures for binarization
            module.temperature_sampling=1e-9
            module.temperature_softplus=1e-9

        if isinstance(module, LogicDense):
            # pick random indeces to set to high values (give priority to certain gates, picked at random)
            n_gates = module.weight.shape[0]
            indices = torch.randint(0, 16, (n_gates,), device=module.weight.device)
            rows = torch.arange(n_gates, device=module.weight.device)

            with torch.no_grad():
                module.weight[rows, indices] = 100

        if isinstance(module, LogicConv2d):
            for layer_weights in module.tree_weights:
                n_kernels, n_inputs = layer_weights.data.shape[:2]
                indices = torch.randint(
                    0, 16,
                    (n_kernels, n_inputs, 1),
                    device=layer_weights.device
                )
                with torch.no_grad():
                    layer_weights.scatter_(2, indices, 100)

    out_train = model(x)

    model.eval()
    out_eval = model(x)
    assert torch.allclose(out_eval, out_train, atol=1e-5), "Eval mode outputs do not match"


@pytest.mark.parametrize("model_class", [DenseModel, ConvModel, BranchModel])
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
