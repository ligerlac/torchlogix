"""Test suite for the base LGN (Logic Gate Network) implementation.

This module contains tests for the core functionality of the LGN class.
"""

from xml.parsers.expat import model
import numpy as np
import pytest
import torch

from torchlogix import CompiledLogicNet
from torchlogix.layers import GroupSum, LogicDense
from torchlogix.functional import take_tuples, walsh_basis_hard

connections_kwargs = {"init_method": "random-unique"}
llkw = {"connections": "fixed", "device": "cpu", "connections_kwargs": connections_kwargs}
llkw_walsh = {"connections": "fixed", "device": "cpu", "parametrization": "walsh", "connections_kwargs": connections_kwargs}
llkw_light = {"connections": "fixed", "device": "cpu", "parametrization": "light", "connections_kwargs": connections_kwargs}



def test_get_luts_and_ids_xor_walsh():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.zeros((1, 4))
    layer.weight.data[0, 3] = 1
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 1, 1, 0]]]))


def test_get_luts_and_ids_and_walsh():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.tensor([[0.5, 0.5, 0.5, -0.5]])
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 0, 0, 1]]]))


def test_regularizer_walsh():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.tensor([[0.5, 0.5, 0.5, -0.5]])
    reg_loss = layer.get_regularization_loss("abs_sum")
    assert np.isclose(reg_loss.item(), 0.0)
    reg_loss = layer.get_regularization_loss("L2")
    assert np.isclose(reg_loss.item(), 0.0)

    layer.weight.data[0] += 1
    reg_loss = layer.get_regularization_loss("abs_sum")
    assert reg_loss.item() > 0.0
    reg_loss = layer.get_regularization_loss("L2")
    assert reg_loss.item() > 0.0


def test_weight_rescale_walsh():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_walsh)
    layer.weight.data = torch.tensor([[0.5, 0.5, 0.5, 1.0]])
    reg_loss = layer.get_regularization_loss("abs_sum")
    assert reg_loss.item() > 0.0
    reg_loss = layer.get_regularization_loss("L2")
    assert reg_loss.item() > 0.0
    layer.rescale_weights("abs_sum")
    reg_loss = layer.get_regularization_loss("abs_sum")
    assert np.isclose(reg_loss.item(), 0.0)
    layer.rescale_weights("L2")
    reg_loss = layer.get_regularization_loss("L2")
    assert np.isclose(reg_loss.item(), 0.0)


def test_get_luts_and_ids_xor_light():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_light)
    layer.weight.data = torch.zeros((1, 4))
    layer.weight.data[0, 1] = 1
    layer.weight.data[0, 2] = 1
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 1, 1, 0]]]))


def test_get_luts_and_ids_and_light():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_light)
    layer.weight.data = torch.tensor([[0, 0, 0, 1.0]])
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 0, 0, 1]]]))
    

def test_get_luts_and_ids_xor():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros((1, 16))
    layer.weight.data[0, 6] = 100
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts, torch.tensor([[[0, 1, 1, 0]]]))


def test_get_luts_and_ids_and():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros((1, 16))
    layer.weight.data[0, 1] = 100
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts, torch.tensor([[[0, 0, 0, 1]]]))


def test_trivial_layer():
    """Test a layer with minimal dimensions.

    Layer w/ 2 inputs and 1 output should have just 1 connection: between 0 and 1
    and its weights should have shape (1, 16).
    It should not be possible to have more than one connection (==out_dim).
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    assert torch.allclose(
        layer.connections.indices, torch.tensor(((0,), (1,)))
        ) or torch.allclose(layer.connections.indices, torch.tensor(((1,), (0,))))
    assert layer.weight.shape == (1, 16)
    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=2, **llkw)


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_trivial_layer_walsh(lut_rank):
    llkw_walsh["connections_kwargs"]["init_method"] = "random"
    layer = LogicDense(in_dim=lut_rank, out_dim=1, lut_rank=lut_rank, **llkw_walsh)
    assert layer.connections.indices.shape == (lut_rank, 1)
    # the connections must be random permutation of all inputs
    assert set(layer.connections.indices[:, 0].tolist()) == set(range(lut_rank))
    assert layer.weight.shape == (1, 2**lut_rank)


@pytest.mark.parametrize("lut_rank", [4, 6])
def test_in_dim_less_than_lut_rank(lut_rank):
    """Test that an error is raised when in_dim < lut_rank."""
    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=1, lut_rank=lut_rank, **llkw_walsh)


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


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_take_tuples(lut_rank):
    llkw_walsh["connections_kwargs"]["init_method"] = "random-unique"
    layer = LogicDense(in_dim=400, out_dim=400, lut_rank=lut_rank, **llkw_walsh)
    # column-wise unique test
    assert all(len(torch.unique(layer.connections.indices[..., col])
                   ) == lut_rank for col in range(layer.connections.indices.shape[1]))
    unique, counts = torch.unique(layer.connections.indices, return_counts=True)
    # counts should not deviate by more than 1
    assert counts.float().std().item() < 1
    # cover all inputs
    assert len(unique) == layer.in_dim


@pytest.mark.parametrize("weight_init", ["random", "residual"])
def test_compiled_model(weight_init):
    """Test model compilation and inference."""
    parametrization_kwargs = {"weight_init": weight_init}
    connections_kwargs = {"init_method": "random"}
    model = torch.nn.Sequential(
        LogicDense(
            in_dim=42,
            out_dim=42,
            connections="fixed",
            connections_kwargs=connections_kwargs,
            parametrization_kwargs=parametrization_kwargs,
            device="cpu",
        ),
        LogicDense(
            in_dim=42,
            out_dim=42,
            connections="fixed",
            connections_kwargs=connections_kwargs,
            parametrization_kwargs=parametrization_kwargs,
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
        LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, device="cpu"),
        LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, device="cpu"),
        LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, device="cpu"),
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


@pytest.mark.parametrize("parametrization", ["raw", "walsh", "light"])
@pytest.mark.parametrize("num_candidates", [-1, 1, 2, 3])
@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_learnable_connections(parametrization, num_candidates, lut_rank):
    """Test that connections can be trained."""
    if lut_rank > 2 and parametrization == "raw":
        pytest.skip("Raw parametrization currently only supports lut_rank=2 ")
    connections_kwargs = {"init_method": "random-unique", "num_candidates": num_candidates}
    in_dim = 100
    out_dim = 100
    layer = LogicDense(in_dim=in_dim, 
                       out_dim=out_dim, 
                       lut_rank=lut_rank, 
                       connections="learnable", 
                       connections_kwargs=connections_kwargs, 
                       device="cpu",
                       parametrization=parametrization)
    if num_candidates == -1:
        assert layer.connections.indices.shape[0] == layer.in_dim
    else:
        assert layer.connections.indices.shape[0] == num_candidates
    assert layer.connections.indices.shape[1] == layer.lut_rank
    assert layer.connections.indices.shape[2] == layer.out_dim
    assert layer.connections.indices.shape == layer.connections.weights.shape
    print(layer.connections.weights.shape)
    X = torch.rand((5, in_dim), requires_grad=True)
    layer.training = True
    y = layer(X)
    loss = y.sum()
    loss.backward()
    assert all(torch.norm(p.grad) > 0 for p in layer.parameters())
    