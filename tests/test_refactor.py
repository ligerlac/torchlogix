import torch
from torchlogix.layers import LogicConv2d
import pytest
import numpy as np


@pytest.mark.parametrize("layer, input_shape", [
    (LogicConv2d(in_dim=32, channels=3, num_kernels=8, tree_depth=2), (1, 3, 32, 32)),
])
def test_individual_layer(layer, input_shape):
    # set all weights to 0 execpt one gate per neuron (randomly selected)
    # this should ensure that the relaxed and discrete outputs match closely
    for param in layer.parameters():
        param.data.zero_()
        random_weight_idx = np.random.randint(0, 16, size=param.shape[:-1])
        leading_idx = tuple(np.arange(s) for s in param.shape[:-1])
        grid = np.ix_(*leading_idx)
        param.data[grid + (random_weight_idx,)] = 100.0

    inp_torch = torch.randint(0, 2, input_shape).float()
    out_relaxed = layer(inp_torch)

    layer.eval()
    out_discrete = layer(inp_torch)

    assert torch.allclose(out_relaxed, out_discrete)
