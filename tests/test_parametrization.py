import torch
from torchlogix.parametrization import (
    RawLUTParametrization, 
    WarpLUTParametrization,
    LightLUTParametrization
)
from torchlogix.functional import ID_TO_WALSH_COEFFICIENTS, walsh_hadamard_transform
from torchlogix.layers import LogicDense
import pytest
import numpy as np
import math


def int_to_bits(n, width=4):
    return [int(b) for b in format(n, f'0{width}b')]


@pytest.mark.parametrize("param_cls, forward_sampling, temperature", [
    (RawLUTParametrization, "hard", 0.5),
    (WarpLUTParametrization, "soft", 1.0),
    (LightLUTParametrization, "gumbel_soft", 2.0),
])
def test_lut_parametrization_init(param_cls, forward_sampling, temperature):
    lut_parametrization = param_cls(
        lut_rank=2,
        forward_sampling=forward_sampling,
        temperature=temperature
    )
    assert lut_parametrization.lut_rank == 2
    assert lut_parametrization.forward_sampling == forward_sampling
    assert lut_parametrization.temperature == temperature


@pytest.mark.parametrize("lut_rank", [4, 6])
def test_raw_lut_rank_fails(lut_rank):
    with pytest.raises(ValueError):
        RawLUTParametrization(
            lut_rank=lut_rank,
            forward_sampling="soft",
            temperature=1.0
        )


@pytest.mark.parametrize("num_neurons", [42, 69])
def test_weight_init_raw(num_neurons):
    param = RawLUTParametrization(lut_rank=2, weight_init="random", residual_probability=1.0)
    weights = param.init_weights(
        num_neurons=num_neurons,
        device="cpu"
    )
    assert weights.shape == (num_neurons, 16)


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
@pytest.mark.parametrize("num_neurons", [42, 69])
@pytest.mark.parametrize("param_cls", [WarpLUTParametrization, LightLUTParametrization])
def test_weight_init_not_raw(lut_rank, num_neurons, param_cls):
    param = param_cls(lut_rank=lut_rank, weight_init="random", residual_probability=0.9)
    weights = param.init_weights(
        num_neurons=num_neurons,
        device="cpu"
    )
    assert weights.shape == (num_neurons, 2**lut_rank)


def test_get_luts_and_ids():
    param = RawLUTParametrization(lut_rank=2)
    for i in range(16):
        weights = torch.zeros((1, 16))
        weights[0, i] = 1.0
        luts, ids = param.get_luts_and_ids(weights)
        assert torch.allclose(ids, torch.tensor([i]))
        # expected lut should be binary representation of i
        expected_lut = torch.tensor(int_to_bits(i, width=4))
        assert torch.allclose(luts, expected_lut)


def test_get_lut_ids_warp():
    param = WarpLUTParametrization(lut_rank=2)
    for i, coeff in ID_TO_WALSH_COEFFICIENTS.items():
        weights = torch.tensor(coeff).float()
        luts, ids = param.get_luts_and_ids(weights)
        assert torch.allclose(ids, torch.tensor([i]))
        inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]])
        inputs = inputs.unsqueeze(2)  # add batch dim
        output = param.forward(x=inputs, training=False, weight=weights)
        # assert output matches truth table (lut)
        assert torch.allclose(output.squeeze(1), luts.float())
        # assert that the lut matches binary representation of i
        assert torch.allclose(luts, torch.tensor(int_to_bits(i, width=4)).bool())


def test_get_lut_ids_light():
    all_coeffs = torch.empty((2**4, 4), dtype=torch.int32)
    for j in range(2**4):
        input = int_to_bits(j, width=4)
        all_coeffs[j] = torch.tensor(input)

    param = LightLUTParametrization(lut_rank=2)
    for i, coeff in enumerate(all_coeffs):
        weights = torch.tensor(coeff).float()
        luts, ids = param.get_luts_and_ids(weights)
        assert torch.allclose(ids, torch.tensor([i]))
        inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]])
        inputs = inputs.unsqueeze(2)  # add batch dim
        output = param.forward(x=inputs, training=False, weight=weights)
        # assert output matches truth table (lut)
        assert torch.allclose(output.squeeze(1), luts.float())
        # assert that the lut matches binary representation of i
        assert torch.allclose(luts, torch.tensor(int_to_bits(i, width=4)).bool())


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_get_luts_warp(lut_rank):
    all_inputs = torch.empty((2**lut_rank, lut_rank), dtype=torch.int32)
    for j in range(2**lut_rank):
        input = int_to_bits(j, width=lut_rank)
        all_inputs[j] = torch.tensor(input)

    param = WarpLUTParametrization(lut_rank=lut_rank)
    for i in range(16):
        input_lut = torch.tensor(int_to_bits(i, width=2**lut_rank))
        warp_input = 1 - 2 * input_lut.unsqueeze(0).float()
        weights = 1./2**lut_rank * walsh_hadamard_transform(warp_input, lut_rank)
        # check L2 norm
        assert torch.allclose(weights.norm(p=2), torch.tensor(1.0))
        output_lut = param.forward(x=all_inputs.unsqueeze(2), training=False, weight=weights)
        assert torch.allclose(output_lut.squeeze(), input_lut.float())


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_get_luts_light(lut_rank):
    all_inputs = torch.empty((2**lut_rank, lut_rank), dtype=torch.int32)
    for j in range(2**lut_rank):
        input = int_to_bits(j, width=lut_rank)
        all_inputs[j] = torch.tensor(input)

    param = LightLUTParametrization(lut_rank=lut_rank)
    for i in range(16):
        input_lut = torch.tensor(int_to_bits(i, width=2**lut_rank))
        weights = input_lut.unsqueeze(0).float()
        # check L2 norm
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        output_lut = param.forward(x=all_inputs.unsqueeze(2), training=False, weight=weights)
        assert torch.allclose(output_lut.squeeze(), input_lut.float())


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
@pytest.mark.parametrize("num_neurons", [42, 69])
@pytest.mark.parametrize("residual_probability", [0.6, 0.8, 0.95, 0.99])
def test_residual_init_warp(lut_rank, num_neurons, residual_probability):
    param = WarpLUTParametrization(lut_rank=lut_rank, weight_init="residual", residual_probability=residual_probability)
    weights = param.init_weights(
        num_neurons=num_neurons,
        device="cpu"
    )
    lut_entries = 2**lut_rank
    identity = 1 - 2 * torch.cat((torch.zeros(lut_entries // 2), torch.ones(lut_entries // 2)))
    assert all(torch.allclose(walsh_hadamard_transform(w / torch.norm(w, p=2), lut_rank), identity) for w in weights)


@pytest.mark.parametrize("num_neurons", [42, 69])
@pytest.mark.parametrize("residual_probability", [0.5, 0.9, 0.95, 0.99])
def test_residual_consistency(num_neurons, residual_probability):
    lut_rank = 2
    parametrization_kwargs = {
        "weight_init": "residual", "residual_probability": residual_probability}
    warp = LogicDense(in_dim=lut_rank, out_dim=num_neurons, lut_rank=lut_rank, parametrization="warp", 
                       parametrization_kwargs=parametrization_kwargs, device="cpu")
    warp.connections.indices = torch.arange(2).unsqueeze(-1)
    raw = LogicDense(in_dim=lut_rank, out_dim=num_neurons, lut_rank=lut_rank, parametrization="raw", 
                       parametrization_kwargs=parametrization_kwargs, device="cpu")
    raw.connections.indices = torch.arange(2).unsqueeze(-1)
    X = (torch.arange(1 << lut_rank)[:, None] >> torch.arange(lut_rank) & 1).flip(-1).to(torch.float32)
    assert torch.allclose(warp(X), raw(X))


def test_raw_train_and_eval_match_for_fixed_luts():
    param = RawLUTParametrization(lut_rank=2, forward_sampling="hard")
    weights = torch.full((3, 16), -10.0)
    weights[0, 0] = 10.0
    weights[1, 6] = 10.0
    weights[2, 15] = 10.0

    x = torch.tensor([
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    ])

    train_out = param.forward(x=x, training=True, weight=weights)
    eval_out = param.forward(x=x, training=False, weight=weights)

    assert torch.allclose(train_out, eval_out)
    

@pytest.mark.parametrize("param_cls", [RawLUTParametrization, WarpLUTParametrization, LightLUTParametrization])
@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_materialize_basis(param_cls, lut_rank):
    if lut_rank > 2 and param_cls == RawLUTParametrization:
        pytest.skip("Raw parametrization not supported for lut_rank > 2")

    if param_cls == RawLUTParametrization:
        weights = torch.randn((10, 16)) 
    else:
        weights = torch.randn((10, 2**lut_rank))

    inputs = torch.randint(0, 2, (8, lut_rank)).unsqueeze(2).float()
    param_in_place = param_cls(lut_rank=lut_rank, materialize_basis=False)
    param_materialize = param_cls(lut_rank=lut_rank, materialize_basis=True)

    output_in_place = param_in_place.forward(
        x=inputs,
        training=True,
        weight=weights,
    )

    output_materialize = param_materialize.forward(
        x=inputs,
        training=True,
        weight=weights,
    )

    assert torch.allclose(output_in_place, output_materialize)
