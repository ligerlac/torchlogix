import pytest

from torchlogix.layers import Binarization, FixedBinarization, LearnableBinarization
import torch


def test_uniform_binarization():
    data = torch.arange(9).float().reshape((3, 3))
    thresholds = Binarization.get_uniform_thresholds(data, num_bits=2, feature_wise=True)
    binarizer = FixedBinarization(thresholds)
    assert torch.allclose(binarizer.thresholds, 
                          torch.tensor([[2, 4], [3, 5], [4, 6]], dtype=torch.float32))
    x = torch.tensor([[2, 4, 4], [6, 5, 8], [3, 7, 5]]).float()
    output = binarizer(x)
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                             [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"

    thresholds = Binarization.get_uniform_thresholds(data, num_bits=2, feature_wise=False)
    binarizer = FixedBinarization(thresholds)
    output = binarizer(x)
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                             [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"


def test_uniform_distributive_binarization():
    data = torch.arange(12).float().reshape((4, 3))
    uniform_thresholds = Binarization.get_uniform_thresholds(data, num_bits=2, feature_wise=True)
    distributive_thresholds = Binarization.get_distributive_thresholds(data, num_bits=2, feature_wise=True)
    assert torch.allclose(uniform_thresholds, distributive_thresholds)

    data[1, :] += 0.5  # make feature 1 distributive different
    uniform_thresholds_ = Binarization.get_uniform_thresholds(data, num_bits=2, feature_wise=True)
    distributive_thresholds_ = Binarization.get_distributive_thresholds(data, num_bits=2, feature_wise=True)
    assert torch.allclose(uniform_thresholds, uniform_thresholds_)
    assert not torch.allclose(uniform_thresholds_, distributive_thresholds_)


@pytest.mark.parametrize("num_bits", [2, 4, 6, 8])
def test_uniform_binarization_image(num_bits):
    channels, width, height = 2, 2, 2
    data = torch.rand(100, channels, width, height)
    thresholds = Binarization.get_uniform_thresholds(data, num_bits=num_bits, feature_wise=True)
    binarizer = FixedBinarization(thresholds, feature_dim=1)
    x = torch.rand(10, channels, width, height)
    output = binarizer(x)
    assert output.shape == (10, channels * num_bits, width, height)  # last dim is num_bits

    thresholds = Binarization.get_uniform_thresholds(data, num_bits=num_bits, feature_wise=False)
    binarizer = FixedBinarization(thresholds, feature_dim=1)
    output = binarizer(x)
    assert output.shape == (10, channels * num_bits, width, height)  # last dim is num_bits


def test_learnable_binarization():
    data = torch.arange(9).float().reshape((3, 3))
    num_bits = 2
    feature_wise = True
    thresholds = Binarization.get_uniform_thresholds(data, num_bits=num_bits, feature_wise=feature_wise)
    binarizer = LearnableBinarization(thresholds, num_bits=num_bits, feature_wise=feature_wise, 
                                      temperature_sampling=0.0001)
    x = torch.tensor([[2, 4, 4], [6, 5, 8], [3, 7, 5]]).float() 
    output = binarizer(x)
    loss = output.sum()
    loss.backward()
    assert binarizer.raw_diffs.grad is not None
    
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                             [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"
    assert binarizer.get_thresholds().requires_grad
    binarizer.freeze_thresholds()
    assert not binarizer.get_thresholds().requires_grad

    feature_wise = False
    thresholds = Binarization.get_uniform_thresholds(data, num_bits=num_bits, feature_wise=feature_wise)
    binarizer = LearnableBinarization(thresholds, num_bits=num_bits, feature_wise=feature_wise, temperature_sampling=0.0001)
    output = binarizer(x)
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                             [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"


@pytest.mark.parametrize("num_bits", [4, 6, 8])
def test_learnable_binarization_image(num_bits):
    channels, width, height = 3, 5, 5
    data = torch.rand(100, channels, width, height)
    thresholds = Binarization.get_uniform_thresholds(data, num_bits=num_bits, feature_wise=True)
    binarizer = LearnableBinarization(thresholds, num_bits=num_bits, feature_wise=True, feature_dim=1)
    x = torch.rand(10, channels, width, height)
    output = binarizer(x)
    assert output.shape == (10, channels * num_bits, width, height)  # last dim is num_bits

    thresholds = Binarization.get_uniform_thresholds(data, num_bits=num_bits, feature_wise=False)
    binarizer = LearnableBinarization(thresholds, num_bits=num_bits, feature_wise=False, feature_dim=1)
    output = binarizer(x)
    assert output.shape == (10, channels * num_bits, width, height)  # last dim is num_bits
