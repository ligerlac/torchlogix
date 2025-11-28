"""Test suite for the CLGN (Convolutional Logic Gate Network) implementation - 3D convolutions.

This module contains tests for the core functionality of the CLGN class.
"""

import math
import pytest
import numpy as np
import torch
from torch.nn.modules.utils import _triple

from torchlogix.layers import LogicConv3d, OrPooling, GroupSum
from torchlogix import CompiledLogicNet


@pytest.fixture
def layer(
    in_dim, channels, num_kernels, tree_depth, receptive_field_size, stride, padding, connections
):
    """Create instance of LogicCNNLayer."""
    params = {
        "in_dim": in_dim,
        "device": "cpu",
        "channels": channels,
        "num_kernels": num_kernels,
        "tree_depth": tree_depth,
        "receptive_field_size": receptive_field_size,
        "implementation": "python",
        "connections": connections,
        "stride": stride,
        "padding": padding,
    }
    receptive_field_size_tuple = _triple(receptive_field_size)
    in_dim_tuple = _triple(in_dim)

    # in_dim can be an integer or a tuple of integers. be m either the int
    # itself or the min of the tuple
    if (
            (receptive_field_size_tuple[0] > in_dim_tuple[0]) or
            (receptive_field_size_tuple[1] > in_dim_tuple[1]) or
            (receptive_field_size_tuple[2] > in_dim_tuple[2])
    ):
        with pytest.raises(AssertionError):
            LogicConv3d(**params)
        pytest.skip("Receptive field size should be smaller than input dimension")

    if stride > min(receptive_field_size_tuple):
        with pytest.raises(AssertionError):
            LogicConv3d(**params)
        pytest.skip("Stride should be smaller than receptive field size")
    kernel_volume = math.prod(receptive_field_size_tuple) * channels
    if connections == "random-unique":
        if kernel_volume * (kernel_volume - 1) / 2 < 2** tree_depth:
            pytest.skip("Kernel volume should be large enough to support the tree depth")
    return LogicConv3d(**params)


@pytest.mark.parametrize("in_dim", [2, 7, (18, 14, 6)])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("num_kernels", [1, 5])
@pytest.mark.parametrize("tree_depth", [1, 3])
@pytest.mark.parametrize("receptive_field_size", [2, 3, (3, 2, 2)])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("connections", ["random", "random-unique"])
class TestIndices:
    """Test the shape and structure of layer indices.

    A layer's indices are of the shape [level_N, level_N-1, ..., level_0], where N is
    the tree depth. Each level is of shape (left_indices, right_indices), defining the
    inputs for the binary logic gates.
    """

    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_first_tree_level_shape(self, layer, side) -> None:
        """Test the shape of the first tree level indices.

        The first tree level defines which entries within the receptive field are
        considered. It should be of shape (num_kernels, num_positions, 2**tree_depth, 4)
        [4 because of (w, h, d, c) notation].
        """
        h_positions = (
            int((layer.in_dim[0] + 2 * layer.padding - layer.receptive_field_size[0]) / layer.stride) + 1
        )
        w_positions = (
            int((layer.in_dim[1] + 2 * layer.padding - layer.receptive_field_size[1]) / layer.stride) + 1
        )
        d_positions = (
            int((layer.in_dim[2] + 2 * layer.padding - layer.receptive_field_size[2]) / layer.stride) + 1
        )
        num_positions = h_positions * w_positions * d_positions
        indices = layer.indices[0][side]
        assert indices.shape == (
            layer.num_kernels,
            num_positions,
            2**layer.tree_depth,
            4,
        )

    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_other_tree_levels_shape(self, layer, side) -> None:
        """Test the shape of other tree level indices.

        Since the convolution is implemented as a binary tree, all following levels
        should have 2**i gates, where i is the level (in reverse order).
        """
        for level in range(1, layer.tree_depth):
            indices = layer.indices[level][side]
            expected_gates = 2 ** (layer.tree_depth - level)
            assert indices.shape == (expected_gates,)


    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_first_tree_level_range(self, layer, side):
        """Test that indices are within input dimensions.

        Width, height and channel indices should be within specified input dimensions.
        """
        indices = layer.indices[0][side]
        assert torch.all(indices[..., 0] < layer.in_dim[0])
        assert torch.all(indices[..., 1] < layer.in_dim[1])
        assert torch.all(indices[..., 2] < layer.in_dim[2])
        assert torch.all(indices[..., 3] < layer.channels)

    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_other_tree_levels_range(self, layer, side):
        """Test that indices are within previous level range.

        Each following level should have indices within the range of the previous level.
        """
        for level in range(1, layer.tree_depth):
            indices = layer.indices[level][side]
            n_gates_prev = 2 ** (layer.tree_depth - level + 1)
            assert torch.all(indices < n_gates_prev)


    def test_uniqueness(self, layer):
        """Test that indices are unique within the first level.
        For random-unique connections, the first level should have unique pairs of
        indices.
        """
        if layer.connections != "random-unique":
            pytest.skip("Test only applies to random-unique connections")

        # Only test the first level (level 0) which contains the actual position pairs
        left_indices = layer.indices[0][0]   # Shape: (num_kernels, num_positions, sample_size, 4)
        right_indices = layer.indices[0][1]  # Shape: (num_kernels, num_positions, sample_size, 4)

        # Test uniqueness for each kernel and each sliding position
        for kernel_idx in range(left_indices.shape[0]):
            for pos_idx in range(left_indices.shape[1]):
                left_pos = left_indices[kernel_idx, pos_idx]    # Shape: (sample_size, 4)
                right_pos = right_indices[kernel_idx, pos_idx]  # Shape: (sample_size, 4)

                # Convert tensor pairs to tuples for comparison
                pairs = []
                for i in range(left_pos.shape[0]):
                    left_tuple = tuple(left_pos[i].tolist())
                    right_tuple = tuple(right_pos[i].tolist())
                    # Ensure consistent ordering (smaller index first) for uniqueness check
                    pair = (left_tuple, right_tuple) if left_tuple < right_tuple else (right_tuple, left_tuple)
                    pairs.append(pair)

                # Check that all pairs are unique
                unique_pairs = set(pairs)
                assert len(unique_pairs) == len(pairs), \
                    f"Kernel {kernel_idx}, position {pos_idx}: Found duplicate pairs. " \
                    f"Expected {len(pairs)} unique pairs, got {len(unique_pairs)}"

                # Also check that no self-connections exist
                for left_tuple, right_tuple in pairs:
                    assert left_tuple != right_tuple, \
                        f"Kernel {kernel_idx}, position {pos_idx}: Found self-connection {left_tuple}"


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
        tree_depth=1,
        receptive_field_size=2,
        implementation="python",
        connections="random-unique",
        stride=1,
        padding=0,
    )

    kernel_pairs = (
        torch.tensor([[[0, 0, 0, 0], [0, 1, 0, 0]]]),
        torch.tensor([[[0, 0, 1, 0], [0, 1, 1, 0]]]),
    )
    layer.indices = layer.get_indices_from_kernel_pairs(kernel_pairs)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0][0].data = and_weights
        layer.tree_weights[0][1].data = and_weights
        layer.tree_weights[1][0].data = and_weights

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
        tree_depth=1,
        receptive_field_size=2,
        implementation="python",
        connections="random-unique",
        stride=1,
        padding=0,
    )

    kernel_pairs = (
        torch.tensor([[[0, 0, 0, 0], [1, 0, 0, 0]]]),
        torch.tensor([[[0, 1, 0, 0], [1, 1, 0, 0]]]),
    )
    layer.indices = layer.get_indices_from_kernel_pairs(kernel_pairs)

    # Set weights to BARELY select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 1.0  # Pick 1 instead of 100 here
        layer.tree_weights[0][0].data = and_weights
        layer.tree_weights[0][1].data = and_weights
        layer.tree_weights[1][0].data = and_weights

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
        print(f"Input: {x}, Output: {output}, Expected: {expected}")
        assert torch.allclose(output, expected)


def test_conv_model():
    layer = LogicConv3d(
        in_dim=3,
        device="cpu",
        channels=1,
        num_kernels=1,
        tree_depth=1,
        receptive_field_size=2,
        implementation="python",
        connections="random-unique",
        stride=1,
        padding=0,
    )

    kernel_pairs = (
        torch.tensor([[[0, 0, 0, 0], [1, 0, 0, 0]]]),
        torch.tensor([[[0, 1, 0, 0], [1, 1, 0, 0]]]),
    )
    layer.indices = layer.get_indices_from_kernel_pairs(kernel_pairs)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0][0].data = and_weights
        layer.tree_weights[0][1].data = and_weights
        layer.tree_weights[1][0].data = and_weights

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
        tree_depth=1,
        receptive_field_size=(3,2,2),
        implementation="python",
        connections="random-unique",
        stride=1,
        padding=0,
    )

    kernel_pairs = (
        torch.tensor([[[0, 0, 0, 0], [1, 0, 0, 0]]]),
        torch.tensor([[[0, 1, 0, 0], [1, 1, 0, 0]]]),
    )
    layer.indices = layer.get_indices_from_kernel_pairs(kernel_pairs)

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0][0].data = and_weights
        layer.tree_weights[0][1].data = and_weights
        layer.tree_weights[1][0].data = and_weights

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
        print("output = ", output)
        expected = torch.tensor(y, dtype=torch.float)
        print("expected = ", expected)
        assert torch.allclose(
            output,
            expected
        )


def test_pooling_layer():
    layer = OrPooling(
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

        print(f"x.shape = {x.shape}")
        output = layer(x)
        expected = y.unsqueeze(0).unsqueeze(0)  # [1, 1, H_out, W_out, D_out]

        print(f"Input: {x}, Output: {output}, Expected: {expected}")
        assert torch.allclose(output, expected)


def test_compiled_model():
    """Test model compilation and inference."""
    model = torch.nn.Sequential(
        LogicConv3d(
            in_dim=3,
            device="cpu",
            channels=1,
            num_kernels=1,
            tree_depth=1,
            receptive_field_size=2,
            implementation="python",
            connections="random-unique",
            stride=1,
            padding=0,
        ),
        torch.nn.Flatten(),
        GroupSum(1),
    )

    model.train(False)  # Switch model to eval mode
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 3, 3, 3), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_conv_model.so", verbose=False)

    # 8 random images of shape (1, 3, 3, 3) (single channel, 3x3x3 input)
    X = torch.randint(0, 2, (8, 1, 3, 3, 3)).float()

    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())

    assert np.allclose(preds, preds_compiled)


def test_compiled_model_rect():
    """Test model compilation and inference."""
    model = torch.nn.Sequential(
        LogicConv3d(
            in_dim=(3,4,3),
            device="cpu",
            channels=1,
            num_kernels=1,
            tree_depth=1,
            receptive_field_size=2,
            implementation="python",
            connections="random-unique",
            stride=1,
            padding=0,
        ),
        torch.nn.Flatten(),
        GroupSum(1),
    )

    model.train(False)  # Switch model to eval mode
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 3, 4, 3), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_conv_model.so", verbose=False)

    # 8 random images of shape (1, 3, 4, 3) (single channel, 3x4x3 input)
    X = torch.randint(0, 2, (8, 1, 3, 4, 3)).float()

    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())

    assert np.allclose(preds, preds_compiled)

def test_compiled_pooling_model():
    """Test model compilation and inference."""
    model = torch.nn.Sequential(
        LogicConv3d(
            in_dim=3,
            device="cpu",
            channels=1,
            num_kernels=1,
            tree_depth=1,
            receptive_field_size=2,
            implementation="python",
            connections="random-unique",
            stride=1,
            padding=0,
        ),
        OrPooling(kernel_size=2, stride=2, padding=0),
        torch.nn.Flatten(),
        GroupSum(1),
    )

    model.train(False)  # Switch model to eval mode
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 3, 3, 3), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_conv_model.so", verbose=False)

    # 8 random images of shape (1, 3, 3, 3) (single channel, 3x3x3 input)
    X = torch.randint(0, 2, (8, 1, 3, 3, 3)).float()

    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())

    assert np.allclose(preds, preds_compiled)
