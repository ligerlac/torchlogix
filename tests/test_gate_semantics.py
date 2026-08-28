"""Gate-level semantics: specific weights and wirings produce specific logic.

These tests pin down *behaviour*, not shapes: each one wires a layer by hand
(or sets saturated weights), feeds a small truth table, and asserts the exact
expected output, or checks that get_luts_and_ids() reports the gates that were
configured.

They are deliberately NOT merged across 2D/3D or conv/dense. Their expected
values are hand-computed against a particular spatial arrangement, so a shared
parametrized version would have to branch on dimension to pick its truth table
and would be harder to read than two explicit tests. The formula-based layer
properties, where merging *does* pay off, live in test_layers.py.
"""
import numpy as np
import pytest
import torch

from torchlogix.functional import take_tuples, walsh_basis_hard
from torchlogix.layers import (
    FixedBinarization,
    GroupSum,
    LearnableBinarization,
    LogicConv2d,
    LogicConv3d,
    LogicDense,
    OrPooling2d,
    OrPooling3d,
)

connections_kwargs = {"init_method": "random-unique"}
llkw = {"connections": "fixed", "device": "cpu", "connections_kwargs": connections_kwargs}
llkw_warp = {**llkw, "parametrization": "warp"}
llkw_light = {**llkw, "parametrization": "light"}



def test_unique_connections_warp_conv2d():
    """Test scaling up to multiple inputs, that is n=4."""
    lut_rank = 6
    import time
    start = time.time()
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=(30, 20),
        parametrization="warp",
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=1,
        receptive_field_size=5,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
        lut_rank=lut_rank,
    )
    assert time.time() - start < 10, "Unique connections generation took too long"


def test_lut_rank_warp_conv2d():
    """Test scaling up to multiple inputs, that is n=4."""
    lut_rank = 4
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=(3, 4),
        parametrization="warp",
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=1,
        receptive_field_size=3,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
        lut_rank=lut_rank,
    )
    luts, ids = layer.get_luts_and_ids()
    for luts_level in luts:
        for luts_ in luts_level:
            assert luts_.shape[-1] == 1 << lut_rank


def test_and_model_conv2d():
    """Test the AND gate implementation.

    AND is the 1-st gate:
    - set the weights to 0, except for the 1-st element (set to some high value)
    - test some possible inputs
    """
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=3,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0], [1, 0, 0]]],
        [[[0, 1, 0], [1, 1, 0]]]
    ])
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
        ([[0, 0, 0], 
          [0, 0, 0], 
          [0, 0, 0]
        ], [0, 0, 0, 0]),
        ([[1, 1, 1], 
          [1, 1, 0], 
          [0, 0, 1]]
        , [1, 0, 0, 0]),
        ([[1, 1, 1], 
          [1, 1, 1], 
          [0, 0, 1]]
        , [1, 1, 0, 0]),
        ([[1, 1, 1], 
          [1, 1, 1], 
          [1, 1, 1]]
        , [1, 1, 1, 1]),
    ]

    for x, y in test_cases:
        x = torch.tensor([[x]], dtype=torch.float32)
        output = layer(x)
        expected = torch.tensor(y, dtype=torch.float32).reshape(1, 1, 2, 2)
        assert torch.allclose(
            output, 
            expected
        )


def test_and_model_warp_conv2d():
    """Test the AND gate implementation.

    AND is the 1-st gate:
    - set the weights to 0, except for the 1-st element (set to some high value)
    - test some possible inputs
    """
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=3,
        parametrization="warp",
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0], [1, 0, 0]]],
        [[[0, 1, 0], [1, 1, 0]]]
    ])
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to select AND operation
    # Correct WARP weights for AND gate with {-1, +1} conversion
    # Basis: [1, B, A, A*B] where inputs are converted via x = 1 - 2*x
    # Sigmoid sampler negates: output = ((-x) > 0).float()
    # Scale weights to make sigmoid outputs sharp (like raw parametrization uses 100.0)
    with torch.no_grad():
        and_weights = torch.tensor([50., 50., 50., -50.]).reshape(1, 4)
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    # only all 1s should produce 1
    test_cases = [
        ([[0, 0, 0], 
          [0, 0, 0], 
          [0, 0, 0]
        ], [0, 0, 0, 0]),
        ([[1, 1, 1], 
          [1, 1, 0], 
          [0, 0, 1]]
        , [1, 0, 0, 0]),
        ([[1, 1, 1], 
          [1, 1, 1], 
          [0, 0, 1]]
        , [1, 1, 0, 0]),
        ([[1, 1, 1], 
          [1, 1, 1], 
          [1, 1, 1]]
        , [1, 1, 1, 1]),
    ]

    for x, y in test_cases:
        x = torch.tensor([[x]], dtype=torch.float32)
        output = layer(x)
        expected = torch.tensor(y, dtype=torch.float32).reshape(1, 1, 2, 2)
        assert torch.allclose(
            output, 
            expected
        )


def test_get_luts_and_ids_conv2d():
    """Test the AND gate implementation.

    AND is the 1-st gate:
    - set the weights to 0, except for the 1-st element (set to some high value)
    - test some possible inputs
    """
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=3,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0], [1, 0, 0]]],
        [[[0, 1, 0], [1, 1, 0]]]
    ])
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(luts[0][0].to(torch.long), torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(luts[0][1].to(torch.long), torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(luts[1][0].to(torch.long), torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(ids[0][0], torch.tensor([1]))
    assert torch.allclose(ids[0][1], torch.tensor([1]))
    assert torch.allclose(ids[1][0], torch.tensor([1]))


def test_get_luts_and_ids_and_warp_conv2d():
    """Test the AND gate implementation.

    AND is the 1-st gate:
    - set the weights to 0, except for the 1-st element (set to some high value)
    - test some possible inputs
    """
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=3,
        parametrization="warp",
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0], [1, 0, 0]]],
        [[[0, 1, 0], [1, 1, 0]]]
    ])
    layer.connections.indices = layer.connections._get_indices_from_kernel_tensor(kernels)

    # Set weights to select AND operation
    # Correct WARP weights for AND gate with {-1, +1} conversion
    with torch.no_grad():
        and_weights = torch.tensor([[50., 50., 50., -50.]])
        layer.tree_weights[0].data[0] = and_weights
        layer.tree_weights[0].data[1] = and_weights
        layer.tree_weights[1].data[0] = and_weights

    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(luts[0][0].to(torch.long), torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(luts[0][1].to(torch.long), torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(luts[1][0].to(torch.long), torch.tensor([[0, 0, 0, 1]]))
    assert torch.allclose(ids[0][0], torch.tensor([1]))
    assert torch.allclose(ids[0][1], torch.tensor([1]))
    assert torch.allclose(ids[1][0], torch.tensor([1]))


def test_binary_model_conv2d():
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=2,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0], [1, 0, 0]]],
        [[[0, 1, 0], [1, 1, 0]]]
    ])
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
        ([[0, 0], 
          [0, 0], 
        ], [0]),
        ([[1, 1], 
          [1, 1], 
        ], [1]),
    ]

    for x, y in test_cases:
        x = torch.tensor([[x]], dtype=torch.float32)
        output = layer(x)
        expected = torch.tensor(y, dtype=torch.float32)
        assert torch.allclose(
            output, 
            expected
        )


def test_conv_model_2d():
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=3,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0], [1, 0, 0]]],
        [[[0, 1, 0], [1, 1, 0]]]
    ])
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
        ([[0, 0, 0], 
          [0, 0, 0], 
          [0, 0, 0]
        ], 0),
        ([[1, 1, 1], 
          [1, 1, 0], 
          [0, 0, 1]]
        , 1),
        ([[1, 1, 1], 
          [1, 1, 1], 
          [0, 0, 1]]
        , 2),
        ([[1, 1, 1], 
          [1, 1, 1], 
          [1, 1, 1]]
        , 4),
    ]

    for x, y in test_cases:
        x = torch.tensor([[x]], dtype=torch.float32)
        output = model(x)
        expected = torch.tensor(y, dtype=torch.float32)
        assert torch.allclose(
            output, 
            expected
        )


def test_conv_model_rect_2d():
    connections_kwargs = {"init_method": "random-unique"}
    layer = LogicConv2d(
        in_dim=(3, 4),
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=2,
        receptive_field_size=2,
        connections_kwargs=connections_kwargs,
        stride=1,
        padding=0,
    )

    kernels = torch.tensor([
        [[[0, 0, 0, 0], [1, 0, 0, 0]]],
        [[[0, 1, 0, 0], [1, 1, 0, 0]]]
    ])
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
        ([[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]
        ], 0),
        ([[1, 1, 1, 1],
          [1, 1, 0, 1],
          [0, 0, 1, 1]]
        , 1),
        ([[1, 1, 1, 1],
          [1, 1, 1, 0],
          [0, 0, 1, 1]]
        , 2),
        ([[1, 1, 1, 1],
          [1, 1, 1, 0],
          [0, 1, 1, 1]]
        , 3),
        ([[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]
        , 6),
    ]

    for x, y in test_cases:
        x = torch.tensor([[x]], dtype=torch.float32)
        output = model(x)
        expected = torch.tensor(y, dtype=torch.float32).reshape(1, 1, -1, 1)
        assert torch.allclose(
            output, 
            expected
        )


def test_pooling_layer_2d():
    layer = OrPooling2d(
        kernel_size=2,
        stride=2,
        padding=0,
    )

    test_cases = [
        ([[0, 0, 0, 0], 
          [0, 0, 0, 0], 
          [0, 0, 0, 0],
          [0, 0, 0, 0]
        ], [0, 0, 0, 0]),
        ([[1, 0, 0, 1], 
          [0, 1, 0, 0], 
          [0, 0, 1, 1],
          [1, 0, 0, 1],
        ], [1, 1, 1, 1]),
        ([[1, 1, 1, 1], 
          [1, 1, 1, 1], 
          [0, 0, 1, 1],
          [0, 0, 1, 1],
        ], [1, 1, 0, 1]),
        ([[1, 1, 1, 1], 
          [1, 1, 1, 1], 
          [1, 1, 1, 1],
          [1, 1, 1, 1]
        ], [1, 1, 1, 1]),
    ]

    for x, y in test_cases:
        x = torch.tensor([[x]], dtype=torch.float32)

        output = layer(x)
        expected = torch.tensor(y, dtype=torch.float32).reshape(1, 1, 2, 2)
        assert torch.allclose(
            output, 
            expected
        )    


# ---------------------------------------------------------------------------
# Transposed convolution (LogicConvTranspose2d)
# ---------------------------------------------------------------------------


def test_and_model_conv3d():
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


def test_binary_model_conv3d():
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


def test_lut_rank_walsh_conv3d():
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


def test_conv_model_3d():
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


def test_conv_model_rect_3d():
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


def test_pooling_layer_3d():
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


def test_get_luts_and_ids_xor_warp_dense():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_warp)
    layer.weight.data = torch.zeros((1, 4))
    layer.weight.data[0, 3] = 1
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 1, 1, 0]]]))


def test_get_luts_and_ids_and_warp_dense():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_warp)
    layer.weight.data = torch.tensor([[0.5, 0.5, 0.5, -0.5]])
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 0, 0, 1]]]))


def test_get_luts_and_ids_xor_light_dense():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_light)
    layer.weight.data = torch.zeros((1, 4))
    layer.weight.data[0, 1] = 1
    layer.weight.data[0, 2] = 1
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 1, 1, 0]]]))


def test_get_luts_and_ids_and_light_dense():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_light)
    layer.weight.data = torch.tensor([[0, 0, 0, 1.0]])
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts.to(torch.long), torch.tensor([[[0, 0, 0, 1]]]))


def test_get_luts_and_ids_xor_dense():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros((1, 16))
    layer.weight.data[0, 6] = 100
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([6]))
    assert torch.allclose(luts, torch.tensor([[[0, 1, 1, 0]]]))


def test_get_luts_and_ids_and_dense():
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros((1, 16))
    layer.weight.data[0, 1] = 100
    luts, ids = layer.get_luts_and_ids()
    assert torch.allclose(ids, torch.tensor([1]))
    assert torch.allclose(luts, torch.tensor([[[0, 0, 0, 1]]]))


def test_xor_model_dense():
    """Test the XOR gate implementation.

    XOR is the 6-th gate:
    - set the weights to 0, except for the 6-th element (set to some high value)
    - test the 4 possible inputs
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw)
    layer.weight.data = torch.zeros(16, dtype=torch.float32)
    layer.weight.data[6] = 100
    model = torch.nn.Sequential(layer)
    test_cases = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (x, y), expected in test_cases:
        assert np.isclose(model(torch.tensor([[x, y]], dtype=torch.float32)).item(), expected)


def test_xor_model_warp_dense():
    """Test the XOR gate implementation.

    XOR is the 6-th gate:
    - set the weights to 0, except for the 6-th element (set to some high value)
    - test the 4 possible inputs
    """
    layer = LogicDense(in_dim=2, out_dim=1, **llkw_warp)
    layer.weight.data = torch.zeros(4)
    layer.weight.data[3] = 100
    model = torch.nn.Sequential(layer)
    test_cases = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (x, y), expected in test_cases:
        pred = model(torch.tensor([[x, y]])).item()
        assert np.isclose(pred, expected)
