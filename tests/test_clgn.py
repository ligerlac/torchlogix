"""Test suite for the CLGN (Convolutional Logic Gate Network) implementation.

This module contains tests for the core functionality of the CLGN class.
"""

import pytest
import numpy as np
import torch

from torchlogix.layers import LogicConv2d, OrPooling2d, GroupSum, FixedBinarization, LearnableBinarization
from torchlogix import CompiledLogicNet


@pytest.fixture
def layer(
    in_dim, channels, num_kernels, tree_depth, receptive_field_size, stride, padding, connections_method, weight_init
):
    """Create instance of LogicCNNLayer."""
    params = {
        "in_dim": in_dim,
        "device": "cpu",
        "channels": channels,
        "num_kernels": num_kernels,
        "tree_depth": tree_depth,
        "receptive_field_size": receptive_field_size,
        "connections_kwargs": {"init_method": connections_method},
        "stride": stride,
        "padding": padding,
        "parametrization_kwargs": {"weight_init": weight_init}
    }
    # in_dim can be an integer or a tuple of integers. be m either the int
    # itself or the min of the tuple
    if receptive_field_size > (min(in_dim) if isinstance(in_dim, tuple) else in_dim):
        with pytest.raises(AssertionError):
            LogicConv2d(**params)
        pytest.skip("Receptive field size should be smaller than input dimension")
    if stride > receptive_field_size:
        with pytest.raises(AssertionError):
            LogicConv2d(**params)
        pytest.skip("Stride should be smaller than receptive field size")
    kernel_volume = receptive_field_size ** 2 * channels
    if connections_method == "random-unique":
        if kernel_volume * (kernel_volume - 1) / 2 < 2** tree_depth:
            # with pytest.raises(AssertionError):
            #     LogicConv2d(**params)
            pytest.skip("Kernel volume should be large enough to support the tree depth")
    return LogicConv2d(**params)


@pytest.mark.parametrize("in_dim", [2, 7, (18, 14)])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("num_kernels", [1, 5])
@pytest.mark.parametrize("tree_depth", [1, 3])
@pytest.mark.parametrize("receptive_field_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("connections_method", ["random", "random-unique"])
@pytest.mark.parametrize("weight_init", ["random", "residual"])
class TestIndeces:
    """Test the shape and structure of layer indices.

    A layer's indices are of the shape [level_N, level_N-1, ..., level_0], where N is
    the tree depth. Each level is of shape (left_indices, right_indices), defining the
    inputs for the binary logic gates.
    """

    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_first_tree_level_shape(self, layer, side) -> None:
        """Test the shape of the first tree level indices.

        The first tree level defines which entries within the receptive field are
        considered. It should be of shape (num_positions, num_kernels, 2**(tree_depth-1), 3)
        [3 because of (w, h, c) notation].

        Note: K and P dimensions were swapped (vs. original implementation) so that forward()
        produces (b, k, s, c, f) instead of (b, k, c, s, f), eliminating permute calls.
        """
        vertical_positions = (
            int(
                (layer.in_dim[0] + 2 * layer.padding - layer.receptive_field_size[0])
                / layer.stride
            ) + 1
        )
        horizontal_positions = (
            int(
                (layer.in_dim[1] + 2 * layer.padding - layer.receptive_field_size[1])
                / layer.stride
            ) + 1
        )
        num_positions = horizontal_positions * vertical_positions
        indices = layer.connections.indices[0][side]
        assert indices.shape == (
            num_positions,
            layer.num_kernels,
            2**(layer.tree_depth-1),
            3,
        )


    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_other_tree_levels_shape(self, layer, side) -> None:
        """Test the shape of other tree level indices.

        Since the convolution is implemented as a binary tree, all following levels
        should have 2**(i-1) gates, where i is the level (in reverse order).
        """
        for level in range(1, layer.tree_depth):
            indices = layer.connections.indices[level][side]
            expected_gates = 2 ** (layer.tree_depth - level - 1)
            assert indices.shape == (expected_gates,)


    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_first_tree_level_range(self, layer, side):
        """Test that indices are within input dimensions.

        Width, height and channel indices should be within specified input dimensions.
        """
        indices = layer.connections.indices[0][side]
        assert torch.all(indices[..., 0] < layer.in_dim[0])
        assert torch.all(indices[..., 1] < layer.in_dim[1])
        assert torch.all(indices[..., 2] < layer.channels)

    
    @pytest.mark.parametrize("side", [0, 1], ids=["left", "right"])
    def test_other_tree_levels_range(self, layer, side):
        """Test that indices are within previous level range.

        Each following level should have indices within the range of the previous level.
        """
        for level in range(1, layer.tree_depth):
            indices = layer.connections.indices[level][side]
            n_gates_prev = 2 ** (layer.tree_depth - level + 1)
            assert torch.all(indices < n_gates_prev)


    def test_uniqueness(self, layer):
        """Test that indices are unique within the first level.
        For random-unique connections, the first level should have unique pairs of
        indices.
        """
        if layer.connections.init_method != "random-unique":
            pytest.skip("Test only applies to random-unique connections")
        
        # Only test the first level (level 0) which contains the actual position pairs
        left_indices = layer.connections.indices[0][0]   # Shape: (num_kernels, num_positions, sample_size, 3)
        right_indices = layer.connections.indices[0][1]  # Shape: (num_kernels, num_positions, sample_size, 3)
        
        # Test uniqueness for each kernel and each sliding position
        for kernel_idx in range(left_indices.shape[0]):
            for pos_idx in range(left_indices.shape[1]):
                left_pos = left_indices[kernel_idx, pos_idx]    # Shape: (sample_size, 3)
                right_pos = right_indices[kernel_idx, pos_idx]  # Shape: (sample_size, 3)
                
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


def test_unique_connections_warp():
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


def test_lut_rank_warp():
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


def test_regularizer_warp():
    lut_rank = 2
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
    for params in layer.parameters():
        params.data = torch.tensor([[[0.5, 0.5, 0.5, -0.5]]])
    reg_loss = layer.get_regularization_loss("abs_sum")
    assert np.isclose(reg_loss.item(), 0.0)
    reg_loss = layer.get_regularization_loss("L2")
    assert np.isclose(reg_loss.item(), 0.0)

    for params in layer.parameters():
        for p in params:
            p.data[0] += 1
    reg_loss = layer.get_regularization_loss("abs_sum")
    assert reg_loss.item() > 0.0
    reg_loss = layer.get_regularization_loss("L2")
    assert reg_loss.item() > 0.0


def test_weight_rescale_warp():
    lut_rank = 2
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
    for params in layer.parameters():
        params.data = torch.tensor([[[0.5, 0.5, 0.5, -1.0]]])
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
    

def test_and_model():
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

def test_get_luts_and_ids():
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
    


def test_get_luts_and_ids_and_warp():
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
    

def test_and_model_warp():
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
        print(f"Input:\n{x},\n Output:\n{output},\n Expected:\n{expected}")
        assert torch.allclose(
            output, 
            expected
        )


def test_binary_model():
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
        print(f"x.shape = {x.shape}")
        output = layer(x)
        expected = torch.tensor(y, dtype=torch.float32)
        print(f"Input: {x}, Output: {output}, Expected: {expected}")
        assert torch.allclose(
            output, 
            expected
        )

    
def test_conv_model():
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


def test_conv_model_rect():
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


def test_compiled_model():
    """Test model compilation and inference."""
    connections_kwargs = {"init_method": "random-unique"}
    model = torch.nn.Sequential(
        LogicConv2d(
            in_dim=3,
            device="cpu",
            channels=1,
            num_kernels=1,
            tree_depth=2,
            receptive_field_size=2,
            connections_kwargs=connections_kwargs,
            stride=1,
            padding=0,
        ),
        torch.nn.Flatten(),
        GroupSum(1),
    )

    model.train(False)  # Switch model to eval mode
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 3, 3), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_conv_model.so", verbose=False)

    # 8 random images of shape (1, 3, 3) (single channel, 3x3 input)
    X = torch.randint(0, 2, (8, 1, 3, 3)).float()

    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())

    # c_code = compiled_model.get_c_code()
    # print("Generated C code:")
    # print(c_code)

    print(f"{preds=}")
    print(f"{preds_compiled=}")

    assert np.allclose(preds, preds_compiled)


def test_compiled_model_rect():
    """Test model compilation and inference."""
    connections_kwargs = {"init_method": "random-unique"}
    model = torch.nn.Sequential(
        LogicConv2d(
            in_dim=(3,4),
            device="cpu",
            channels=1,
            num_kernels=1,
            tree_depth=2,
            receptive_field_size=2,
            connections_kwargs=connections_kwargs,
            stride=1,
            padding=0,
        ),
        torch.nn.Flatten(),
        GroupSum(1),
    )

    model.train(False)  # Switch model to eval mode
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 3, 4), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_conv_model.so", verbose=False)

    # 8 random images of shape (1, 3, 4) (single channel, 3x4 input)
    X = torch.randint(0, 2, (8, 1, 3, 4)).float()

    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())

    assert np.allclose(preds, preds_compiled)


def test_pooling_layer():
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

        print(f"x.shape = {x.shape}")
        output = layer(x)
        expected = torch.tensor(y, dtype=torch.float32).reshape(1, 1, 2, 2)
        print(f"Input: {x}, Output: {output}, Expected: {expected}")
        assert torch.allclose(
            output, 
            expected
        )    


def test_compiled_pooling_model():
    """Test model compilation and inference."""
    connections_kwargs = {"init_method": "random-unique"}
    model = torch.nn.Sequential(
        LogicConv2d(
            in_dim=3,
            device="cpu",
            channels=1,
            num_kernels=1,
            tree_depth=2,
            receptive_field_size=2,
            connections_kwargs=connections_kwargs,
            stride=1,
            padding=0,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        torch.nn.Flatten(),
        GroupSum(1),
    )

    model.train(False)  # Switch model to eval mode
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 3, 3), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_conv_model.so", verbose=False)

    # 8 random images of shape (1, 3, 3) (single channel, 3x3 input)
    X = torch.randint(0, 2, (8, 1, 3, 3)).float()

    preds = model(X)
    preds_compiled = compiled_model(X.bool().numpy())

    # c_code = compiled_model.get_c_code()
    # print("Generated C code:")
    # print(c_code)

    assert np.allclose(preds, preds_compiled)


def test_compiled_model_with_thresholding():
    """Test model compilation and inference with FixedBinarization."""
    # Create thresholding layer with frozen thresholds
    thresholds = [10, 20, 30, 40, 50]  # 5 thresholds
    thresholding_layer = FixedBinarization(thresholds=thresholds, one_per="global", feature_dim=1)
    
    # Create a simple model: Thresholding -> Conv -> Flatten -> GroupSum
    connections_kwargs = {"init_method": "random"}
    model = torch.nn.Sequential(
        thresholding_layer,
        LogicConv2d(
            in_dim=3,
            device="cpu",
            channels=5,  # Must match number of thresholds
            num_kernels=2,
            tree_depth=2,
            receptive_field_size=2,
            connections_kwargs=connections_kwargs,
            stride=1,
            padding=0,
        ),
        torch.nn.Flatten(),
        GroupSum(2),  # 2 classes
    )

    model.train(False)  # Switch model to eval mode

    # Test with integer input (will be thresholded)
    X = torch.randint(0, 60, (8, 1, 3, 3)).float()  # 8 samples of shape (1, 3, 3)

    # Get predictions from original model
    preds = model(X)

    # Compile the model (no need to specify apply_thresholding - auto-detected!)
    compiled_model = CompiledLogicNet(
        model=model,
        input_shape=(3, 3),  # Input shape BEFORE thresholding
        num_bits=1,  # Thresholding requires num_bits=1
        cpu_compiler="gcc",
        verbose=True,
        use_bitpacking=True
        # use_bitpacking=False
    )
    compiled_model.compile(save_lib_path="compiled_thresholding_model.so", verbose=False)

    # Get predictions from compiled model
    # Note: compiled model expects integer input (will be thresholded internally)
    preds_compiled = compiled_model(X.int().numpy())

    print(f"Original predictions:\n{preds}")
    print(f"Compiled predictions:\n{preds_compiled}")

    # Verify outputs match
    assert np.allclose(preds.numpy(), preds_compiled, atol=1e-5), \
        "Compiled model with thresholding predictions do not match original model"
    

def test_compiled_model_with_thresholding_float():
    """Test model compilation and inference with FixedBinarization."""
    # Create thresholding layer with frozen thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5 thresholds
    thresholding_layer = FixedBinarization(thresholds=thresholds, one_per="global", feature_dim=1)
    
    # Create a simple model: Thresholding -> Conv -> Flatten -> GroupSum
    connections_kwargs = {"init_method": "random"}
    model = torch.nn.Sequential(
        thresholding_layer,
        LogicConv2d(
            in_dim=3,
            device="cpu",
            channels=5,  # Must match number of thresholds
            num_kernels=2,
            tree_depth=2,
            receptive_field_size=2,
            connections_kwargs=connections_kwargs,
            stride=1,
            padding=0,
        ),
        torch.nn.Flatten(),
        GroupSum(2, tau=2),  # 2 classes
    )

    model.train(False)  # Switch model to eval mode

    # Test with float input (will be thresholded)
    X = torch.rand(8, 1, 3, 3).float()  # 8 samples of shape (1, 3, 3)

    # Get predictions from original model
    preds = model(X)

    # Compile the model (no need to specify apply_thresholding - auto-detected!)
    compiled_model = CompiledLogicNet(
        model=model,
        input_shape=(3, 3),  # Input shape BEFORE thresholding
        num_bits=1,  # Thresholding requires num_bits=1
        cpu_compiler="gcc",
        verbose=True,
        use_bitpacking=True
        # use_bitpacking=False
    )
    compiled_model.compile(save_lib_path="compiled_thresholding_model.so", verbose=False)

    # Get predictions from compiled model
    # Note: compiled model expects integer input (will be thresholded internally)
    preds_compiled = compiled_model(X.numpy())
    print(f"Original predictions:\n{preds}")
    print(f"Compiled predictions:\n{preds_compiled}")

    # Verify outputs match
    assert np.allclose(preds.numpy(), preds_compiled, atol=1e-5), \
        "Compiled model with thresholding predictions do not match original model"
