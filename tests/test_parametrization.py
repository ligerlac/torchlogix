import torch
from torchlogix.parametrization import LUTParametrization, RawLUTParametrization, WalshLUTParametrization
from torchlogix.functional import ID_TO_WALSH_COEFFICIENTS, walsh_hadamard_transform
import pytest


def int_to_bits(n, width=4):
    return [int(b) for b in format(n, f'0{width}b')]


@pytest.mark.parametrize("param_cls, forward_sampling, temperature", [
    (RawLUTParametrization, "hard", 0.5),
    (WalshLUTParametrization, "soft", 1.0),
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
    param = RawLUTParametrization(lut_rank=2)
    weights = param.init_weights(
        weight_init="random",
        num_neurons=num_neurons,
        residual_init_param=0.3,
        device="cpu"
    )
    assert weights.shape == (num_neurons, 16)


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
@pytest.mark.parametrize("num_neurons", [42, 69])
def test_weight_init_walsh(lut_rank, num_neurons):
    param = WalshLUTParametrization(lut_rank=lut_rank)
    weights = param.init_weights(
        weight_init="random",
        num_neurons=num_neurons,
        residual_init_param=0.3,
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


def test_get_lut_ids_walsh():
    param = WalshLUTParametrization(lut_rank=2)
    for i, coeff in ID_TO_WALSH_COEFFICIENTS.items():
        weights = torch.tensor(coeff).float()
        luts, ids = param.get_luts_and_ids(weights)
        assert torch.allclose(ids, torch.tensor([i]))
        inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]])
        inputs = inputs.unsqueeze(2)  # add batch dim
        output = param.forward(x=inputs, training=False, weight=weights, contraction='bnk,nk->bn')
        # assert output matches truth table (lut)
        assert torch.allclose(output.squeeze(1), luts.float())
        # assert that the lut matches binary representation of i
        assert torch.allclose(luts, torch.tensor(int_to_bits(i, width=4)).bool())


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_get_luts_walsh(lut_rank):
    all_inputs = torch.empty((2**lut_rank, lut_rank), dtype=torch.int32)
    for j in range(2**lut_rank):
        input = int_to_bits(j, width=lut_rank)
        all_inputs[j] = torch.tensor(input)

    param = WalshLUTParametrization(lut_rank=lut_rank)
    for i in range(16):
        input_lut = torch.tensor(int_to_bits(i, width=2**lut_rank))
        walsh_input = 1 - 2 * input_lut.unsqueeze(0).float()
        weights = 1./2**lut_rank * walsh_hadamard_transform(walsh_input, lut_rank)
        # check L2 norm
        assert torch.allclose(weights.norm(p=2), torch.tensor(1.0))
        output_lut = param.forward(x=all_inputs.unsqueeze(2), training=False, weight=weights, contraction='bnk,nk->bn')
        assert torch.allclose(output_lut.squeeze(), input_lut.float())
