"""Test suite for the CLGN (Convolutional Logic Gate Network) implementation - 3D convolutions.

This module contains tests for the core functionality of the CLGN class.
"""

import math
import pytest
import numpy as np
import torch
from torch.nn.modules.utils import _triple

from torchlogix.layers import LogicConv3d, LogicConvTranspose3d, OrPooling3d, GroupSum


def test_and_model():
    """Test the AND gate implementation.

    AND is the 1-st gate:
    - set the weights to 0, except for the 1-st element (set to some high value)
    - test some possible inputs
    """
    layer = LogicConv3d(
        in_dim=3,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs={"init_method": "random-unique"},
        stride=1,
        padding=0,
    )

    kernels = torch.tensor(
        [
        [[[0, 0, 0, 0], [0, 1, 0, 0]]],
        [[[0, 0, 1, 0], [0, 1, 1, 0]]],
        ]
    )
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    # only all 1s should produce 1
    test_cases = [
        (torch.zeros((3, 3, 3)), torch.zeros((2, 2, 2))),
        (
            torch.tensor(
                [[[1, 1, 1],
                  [1, 1, 0],
                  [0, 0, 1]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]]
            ),
            torch.tensor(
                [[[1, 0],
                  [0, 0]],
                 [[0, 0],
                  [0, 0]]]
            ),
        ),
        (torch.ones((3, 3, 3)), torch.ones((2, 2, 2))),
    ]

    for x, y in test_cases:
        # Input shape: (batch, channels, H, W, D)
        x = x.unsqueeze(0).unsqueeze(0).float()
        output = layer(x)
        expected = y.unsqueeze(0).unsqueeze(0).float()
        assert torch.allclose(
            output,
            expected
        )

def test_binary_model():
    layer = LogicConv3d(
        in_dim=2,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs={"init_method": "random-unique"},
        stride=1,
        padding=0,
    )

    kernels = torch.tensor(
        [[[[0, 0, 0, 0], [1, 0, 0, 0]]],
        [[[0, 1, 0, 0], [1, 1, 0, 0]]]],
    )
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to BARELY select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 1.0  # Pick 1 instead of 100 here
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    layer.train(False)  # Switch model to eval mode

    test_cases = [
        (
            torch.zeros((2, 2, 2)),  # all zeros input
            torch.zeros((1, 1, 1, 1, 1))   # output should be zero
        ),
        (
            torch.ones((2, 2, 2)),   # all ones input
            torch.ones((1, 1, 1, 1, 1))    # output should be one
        ),
    ]

    for x, y in test_cases:
        # Input shape: (batch, channels, H, W, D)
        x = x.unsqueeze(0).unsqueeze(0).float()
        output = layer(x)
        expected = y.float()
        assert torch.allclose(output, expected)


def test_lut_rank_walsh():
    """Test scaling up to multiple inputs, that is n=4."""
    lut_rank = 4
    layer = LogicConv3d(
        in_dim=(3, 4, 3),
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=1,
        receptive_field_size=3,
        connections_kwargs={"init_method": "random-unique"},
        parametrization="warp",
        stride=1,
        padding=0,
        lut_rank=lut_rank,
    )
    luts, ids = layer.get_luts_and_ids()
    for luts_level in luts:
        for luts_ in luts_level:
            assert luts_.shape[-1] == 1 << lut_rank


def test_conv_model():
    layer = LogicConv3d(
        in_dim=3,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs={"init_method": "random-unique"},
        stride=1,
        padding=0,
    )

    kernels = torch.tensor(
        [[[[0, 0, 0, 0], [1, 0, 0, 0]]],
        [[[0, 1, 0, 0], [1, 1, 0, 0]]]],
    )
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    model = torch.nn.Sequential(layer, torch.nn.Flatten(), GroupSum(1))

    # only all 1s should produce 1
    test_cases = [
        (torch.zeros((3, 3, 3)), 0),
        (
            torch.tensor(
                [[[1, 0, 0],   # (0,0,0)
                  [1, 0, 0],   # (0,1,0)
                  [0, 0, 0]],
                 [[1, 0, 0],   # (1,0,0)
                  [1, 0, 0],   # (1,1,0)
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]]
            ),
            1,
        ),
        (
            torch.tensor(
                [[[1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]],
                 [[1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]]
            ),
            3,
        ),
        (torch.ones((3, 3, 3)), 8),
    ]

    for x, y in test_cases:
        x = x.unsqueeze(0).unsqueeze(0).float()
        output = model(x)
        expected = torch.tensor(y, dtype=torch.float)
        assert torch.allclose(
            output,
            expected
        )

def test_conv_model_rect():
    layer = LogicConv3d(
        in_dim=(4,3,3),
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=(3,2,2),
        connections_kwargs={"init_method": "random-unique"},
        stride=1,
        padding=0,
    )

    kernels = torch.tensor(
        [[[[0, 0, 0, 0], [1, 0, 0, 0]]],
        [[[0, 1, 0, 0], [1, 1, 0, 0]]]],
    )
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    model = torch.nn.Sequential(layer, torch.nn.Flatten(), GroupSum(1))

    # only all 1s should produce 1
    test_cases = [
        (torch.zeros((4, 3, 3)), 0),
        (
            torch.tensor(
                [[[1, 0, 0],
                  [1, 0, 0],
                  [0, 0, 0]],
                 [[1, 0, 0],
                  [1, 0, 0],
                  [0, 0, 0]],
                 [[1, 0, 0],
                  [1, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]]
            ),
            2,
        ),
        (
            torch.tensor(
                [[[1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]],
                 [[1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]],
                 [[1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]]
            ),
            6,
        ),
        (torch.ones((4, 3, 3)), 8),
    ]

    for x, y in test_cases:
        x = x.unsqueeze(0).unsqueeze(0).float()
        output = model(x)
        expected = torch.tensor(y, dtype=torch.float)
        assert torch.allclose(
            output,
            expected
        )


def test_pooling_layer():
    layer = OrPooling3d(
        kernel_size=2,
        stride=2,
        padding=0,
    )

    test_cases = [
        # all zeros
        (
            torch.zeros((4, 4, 4), dtype=torch.float32),
            torch.zeros((2, 2, 2), dtype=torch.float32),
        ),
        # identity along depth slices
        (
            torch.tensor(
                [[[1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [1, 0, 0, 1]],   # depth=0

                 [[0, 1, 0, 0],
                  [1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0]],   # depth=1

                 [[1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [1, 0, 1, 0],
                  [0, 1, 1, 1]],   # depth=2

                 [[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [1, 1, 1, 1]],   # depth=3
                ],
                dtype=torch.float32,
            ),
            torch.ones((2, 2, 2), dtype=torch.float32),  # OR pooling should produce all 1s
        ),
        # all ones
        (
            torch.ones((4, 4, 4), dtype=torch.float32),
            torch.ones((2, 2, 2), dtype=torch.float32),
        ),
    ]

    for x, y in test_cases:
        # Add batch + channel dims: [1, 1, H, W, D]
        x = x.unsqueeze(0).unsqueeze(0)

        output = layer(x)
        expected = y.unsqueeze(0).unsqueeze(0)  # [1, 1, H_out, W_out, D_out]

        assert torch.allclose(output, expected)


# ---------------------------------------------------------------------------
# Transposed convolution (LogicConvTranspose3d)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stride,output_padding,padding", [
    (1, 0, 0), (2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 2, 0),
    (1, 0, 1), (2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 2, 1),
])
def test_conv_transpose3d_output_shape(stride, output_padding, padding):
    """LogicConvTranspose3d must produce the correct spatial output shape."""
    in_d, in_h, in_w = 4, 4, 4
    kH = 3
    channels = 2
    num_kernels = 4
    batch = 2

    if padding > kH - 1:
        pytest.skip("padding must be <= receptive_field_size - 1")

    layer = LogicConvTranspose3d(
        in_dim=(in_d, in_h, in_w),
        channels=channels,
        num_kernels=num_kernels,
        tree_depth=2,
        receptive_field_size=kH,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        device="cpu",
    )

    x = torch.rand(batch, channels, in_d, in_h, in_w)
    out = layer(x)

    expected = (in_d - 1) * stride - 2 * padding + kH + output_padding
    assert out.shape == (batch, num_kernels, expected, expected, expected), (
        f"Expected shape {(batch, num_kernels, expected, expected, expected)}, "
        f"got {tuple(out.shape)}"
    )


def test_conv_transpose3d_gradients():
    """Gradients must flow through LogicConvTranspose3d to all tree-weight parameters."""
    layer = LogicConvTranspose3d(
        in_dim=(4, 4, 4),
        channels=2,
        num_kernels=4,
        tree_depth=2,
        receptive_field_size=3,
        stride=2,
        padding=0,
        output_padding=0,
        parametrization="warp",
        device="cpu",
    )
    layer.train()

    x = torch.rand(2, 2, 4, 4, 4)
    out = layer(x)
    out.sum().backward()

    for i, w in enumerate(layer.tree_weights):
        assert w.grad is not None, f"tree_weights[{i}] has no gradient"
        assert w.grad.abs().sum() > 0, f"tree_weights[{i}] gradient is all zero"


@pytest.mark.parametrize("stride,in_d", [(1, 6), (2, 5), (3, 4)])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv_transpose3d_shape_inverse_of_conv3d(stride, in_d, padding):
    """LogicConvTranspose3d must invert the spatial dimensions of LogicConv3d."""
    kH = 3
    channels = 2
    num_kernels = 4
    batch = 2

    if padding > kH - 1:
        pytest.skip("padding must be <= receptive_field_size - 1")

    conv = LogicConv3d(
        in_dim=(in_d, in_d, in_d),
        channels=channels,
        num_kernels=num_kernels,
        tree_depth=2,
        receptive_field_size=kH,
        stride=stride,
        padding=padding,
        device="cpu",
    )

    x = torch.rand(batch, channels, in_d, in_d, in_d)
    conv_out = conv(x)
    out_d = conv_out.shape[2]

    output_padding = (in_d + 2 * padding - kH) % stride

    transpose_conv = LogicConvTranspose3d(
        in_dim=(out_d, out_d, out_d),
        channels=num_kernels,
        num_kernels=channels,
        tree_depth=2,
        receptive_field_size=kH,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        device="cpu",
    )

    reconstructed = transpose_conv(conv_out)
    assert reconstructed.shape == (batch, channels, in_d, in_d, in_d), (
        f"Expected shape {(batch, channels, in_d, in_d, in_d)}, "
        f"got {tuple(reconstructed.shape)}"
    )


def test_conv_transpose3d_dilation_pads_all_three_axes():
    """Every spatial axis must be dilated - a 2D-shaped pad would leave depth alone."""
    layer = LogicConvTranspose3d(
        in_dim=(4, 4, 4), channels=1, num_kernels=1, tree_depth=1,
        receptive_field_size=2, stride=2, padding=1, output_padding=0,
        device="cpu",
    )
    x = torch.rand(1, 1, 4, 4, 4)
    dilated = layer.connections._make_dilated_input(x)
    assert tuple(dilated.shape) == (1, 1, 7, 7, 7), tuple(dilated.shape)
    # original samples land on even indices, interleaved positions are zero
    assert torch.equal(dilated[:, :, ::2, ::2, ::2], x)
    assert dilated[:, :, 1, 1, 1].abs().sum() == 0


# ---------------------------------------------------------------------------
# Autoencoder: LogicConv3d downsamples, LogicConvTranspose3d reconstructs
# ---------------------------------------------------------------------------

def _make_conv_ae_3d(parametrization="raw"):
    """Small 3D logic autoencoder: 6^3 -> 2^3 (encoder) -> 6^3 (decoder)."""
    encoder = LogicConv3d(
        in_dim=6, channels=2, num_kernels=3, receptive_field_size=3,
        tree_depth=2, stride=2, padding=0, device="cpu",
        parametrization=parametrization,
        parametrization_kwargs={"weight_init": "random"},
    )
    decoder = LogicConvTranspose3d(
        in_dim=2, channels=3, num_kernels=2, receptive_field_size=3,
        tree_depth=2, stride=2, padding=0, output_padding=1, device="cpu",
        parametrization=parametrization,
        parametrization_kwargs={"weight_init": "random"},
    )
    return torch.nn.Sequential(encoder, decoder)


def test_conv_ae_3d_roundtrips_input_shape():
    """The 3D decoder must restore the encoder's input resolution."""
    model = _make_conv_ae_3d()
    x = torch.rand(2, 2, 6, 6, 6)

    encoded = model[0](x)
    assert tuple(encoded.shape) == (2, 3, 2, 2, 2), tuple(encoded.shape)

    decoded = model(x)
    assert tuple(decoded.shape) == (2, 2, 6, 6, 6), (
        f"AE must return to the input resolution, got {tuple(decoded.shape)}"
    )


def test_conv_ae_3d_trains_end_to_end():
    """A reconstruction loss must produce gradients in encoder and decoder alike."""
    torch.manual_seed(0)
    model = _make_conv_ae_3d(parametrization="warp")
    model.train()

    x = (torch.rand(2, 2, 6, 6, 6) > 0.5).float()
    loss = torch.nn.functional.mse_loss(model(x), x)
    loss.backward()

    for name, layer in (("encoder", model[0]), ("decoder", model[1])):
        for i, w in enumerate(layer.tree_weights):
            assert w.grad is not None, f"{name}.tree_weights[{i}] has no gradient"
            assert w.grad.abs().sum() > 0, f"{name}.tree_weights[{i}] gradient is all zero"
