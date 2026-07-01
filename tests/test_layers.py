import torch
import pytest
from torchlogix.layers import LogicDense, GroupSum, LogicConv2d, LogicConv3d, OrPooling2d, OrPooling3d, \
    FixedBinarization, LearnableBinarization, DummyBinarization, Binarization, SoftBinarization


@pytest.mark.parametrize("connections", ["fixed", "learnable"])
@pytest.mark.parametrize("parametrization", ["raw", "warp", "light"])
def test_dense_layer_shapes(connections, parametrization):
    """
    Assure that dense layers use only the last dim as input features
    and broadcast the rest of the dimensions. Meaning: reshaping
    should deliver same result as for-loop + concat.
    """
    layer = LogicDense(in_dim=8, out_dim=8, connections=connections, parametrization=parametrization)
    input = torch.rand((2, 3, 4, 8))
    out = layer(input)
    input_re = input.view(-1, 8)
    out_re = layer(input_re)
    out_re_re = out_re.view(2, 3, 4, 8)
    assert torch.allclose(out, out_re_re)
