"""Test suite for the base LGN (Logic Gate Network) implementation.

This module contains tests for the core functionality of the LGN class.
"""

import numpy as np
import pytest
import torch

from torchlogix import CompiledLogicNet
from torchlogix.layers import GroupSum, LogicDense

llkw = {"connections": "random-unique", "device": "cpu"}
llkw_walsh = {"connections": "random-unique", "device": "cpu", "parametrization": "walsh"}


def test_get_lut_ids_xor_walsh():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.zeros((1, 4))
    layer.weight.data[0, 3] = 1
    luts, ids = layer.get_lut_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 1, 1, 0]]]))


def test_get_lut_ids_and_walsh():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.tensor([[0.5, 0.5, 0.5, -0.5]])
    luts, ids = layer.get_lut_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 0, 0, 1]]]))
    

def test_get_lut_ids_xor():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros((1, 16))
    layer.weight.data[0, 6] = 100
    luts, ids = layer.get_lut_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts, torch.tensor([[[0, 1, 1, 0]]]))


def test_get_lut_ids_and():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros((1, 16))
    layer.weight.data[0, 1] = 100
    luts, ids = layer.get_lut_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts, torch.tensor([[[0, 0, 0, 1]]]))


def test_trivial_layer():
    """Test a layer with minimal dimensions.

    Layer w/ 2 inputs and 1 output should have just 1 connection: between 0 and 1
    and its weights should have shape (1, 16).
    It should not be possible to have more than one connection (==out_dim).
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    assert torch.allclose(layer.indices, torch.tensor(((0,), (1,)))) or torch.allclose(layer.indices, torch.tensor(((1,), (0,))))
    assert layer.weight.shape == (1, 16)
    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=2, **llkw)


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_trivial_layer_walsh(lut_rank):
    layer = LogicDense(in_dim=lut_rank, out_dim=1, lut_rank=lut_rank, parametrization="walsh", connections="random", device="cpu")
    assert layer.indices.shape == (lut_rank, 1)
    # the connections must be random permutation of all inputs
    assert set(layer.indices[:, 0].tolist()) == set(range(lut_rank))
    assert layer.weight.shape == (1, 2**lut_rank)


@pytest.mark.parametrize("lut_rank", [4, 6])
def test_in_dim_less_than_lut_rank(lut_rank):
    """Test that an error is raised when in_dim < lut_rank."""
    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=1, lut_rank=lut_rank, parametrization="walsh", connections="random", device="cpu")


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
        assert np.isclose(model(torch.tensor([[x, y]])).item(), expected)


def test_xor_model_walsh():
    """Test the XOR gate implementation.

    XOR is the 6-th gate:
    - set the weights to 0, except for the 6-th element (set to some high value)
    - test the 4 possible inputs
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.zeros(4)
    layer.weight.data[3] = 100
    model = torch.nn.Sequential(layer)
    test_cases = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (x, y), expected in test_cases:
        pred = model(torch.tensor([[x, y]])).item()
        assert np.isclose(pred, expected)


def test_lut_rank_walsh():
    """Test scaling up to multiple inputs, that is n=4."""
    x = 1 - 2 * torch.rand((1, 12))
    lut_rank = 4
    out_dim = x.shape[1] // lut_rank
    layer = LogicDense(in_dim=x.shape[1], out_dim=out_dim, lut_rank=lut_rank, **llkw_walsh)
    luts, ids = layer.get_lut_ids()
    assert luts.shape == (out_dim, 1 << lut_rank)
    model = torch.nn.Sequential(layer)
    y = model(x)
    assert y.shape == (x.shape[0], out_dim)


def test_compiled_model():
    """Test model compilation and inference."""
    model = torch.nn.Sequential(
        LogicDense(
            in_dim=42,
            out_dim=42,
            connections="random",
            device="cpu",
        ),
        LogicDense(
            in_dim=42,
            out_dim=42,
            connections="random",
            device="cpu",
        ),
        GroupSum(1),
    )
    compiled_model = CompiledLogicNet(
        model=model, num_bits=8, cpu_compiler="gcc", verbose=True
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
        LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, device="cpu"),
        LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, device="cpu"),
        LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, device="cpu"),
        GroupSum(8),
    )
    compiled_model = CompiledLogicNet(
        model=model, num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="minimal_example.so", verbose=False)

    # switch model to eval mode
    model.train(False)

    X = torch.randint(0, 2, (8, 81 * k_num)).int()
    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())
    assert np.allclose(preds, preds_compiled)
