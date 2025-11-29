"""Test suite for the base LGN (Logic Gate Network) implementation.

This module contains tests for the core functionality of the LGN class.
"""

import numpy as np
import pytest
import torch

from torchlogix import CompiledLogicNet
from torchlogix.layers import GroupSum, LogicDense

llkw = {"connections": "unique", "implementation": "python", "device": "cpu"}


def test_trivial_layer():
    """Test a layer with minimal dimensions.

    Layer w/ 2 inputs and 1 output should have just 1 connection: between 0 and 1
    and its weights should have shape (1, 16).
    It should not be possible to have more than one connection (==out_dim).
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    assert layer.indices == (0, 1) or layer.indices == (1, 0)
    assert layer.weight.shape == (1, 16)
    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=2, **llkw)


def test_xor_model():
    """Test the XOR gate implementation.

    XOR is the 6-th gate:
    - set the weights to 0, except for the 6-th element (set to some high value)
    - test the 4 possible inputs
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros(16)
    layer.weight.data[6] = 100
    model = torch.nn.Sequential(layer)
    test_cases = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (x, y), expected in test_cases:
        assert np.isclose(model(torch.tensor([x, y])).item(), expected)


def test_xor_model_walsh():
    """Test the XOR gate implementation.

    XOR is the 6-th gate:
    - set the weights to 0, except for the 6-th element (set to some high value)
    - test the 4 possible inputs
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw, parametrization="walsh")
    layer.weight.data = torch.zeros(4)
    layer.weight.data[3] = -100
    model = torch.nn.Sequential(layer)
    test_cases = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (x, y), expected in test_cases:
        pred = model(torch.tensor([x, y])).item()
        assert np.isclose(pred, expected)


@pytest.mark.parametrize("weight_init", ["random", "residual"])
def test_compiled_model(weight_init):
    """Test model compilation and inference."""
    model = torch.nn.Sequential(
        LogicDense(
            in_dim=42,
            out_dim=42,
            connections="unique",
            weight_init=weight_init,
            implementation="python",
            device="cpu",
        ),
        LogicDense(
            in_dim=42,
            out_dim=42,
            connections="unique",
            weight_init=weight_init,
            implementation="python",
            device="cpu",
        ),
        GroupSum(1),
    )
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(42,), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="minimal_example.so", verbose=False)

    # switch model to eval mode
    model.train(False)

    X = torch.randint(0, 2, (8, 42)).int()
    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())
    assert np.allclose(preds, preds_compiled)


def test_large_compiled_model():
    """Test model compilation and inference."""
    k_num = 16
    model = torch.nn.Sequential(
        LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, device="cpu", implementation="python"),
        LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, device="cpu", implementation="python"),
        LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, device="cpu", implementation="python"),
        GroupSum(8),
    )
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(81 * k_num,), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="minimal_example.so", verbose=False)

    # switch model to eval mode
    model.train(False)

    X = torch.randint(0, 2, (8, 81 * k_num)).int()
    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())
    assert np.allclose(preds, preds_compiled)
