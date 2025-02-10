import pytest
import torch
from difflogic import LogicCNNLayer


@pytest.fixture
def layer(in_dim, channels, num_kernels, tree_depth, receptive_field_size, stride, padding):
    """Create instnace of LogicCNNLayer"""
    params = {
        'in_dim': in_dim,
        'device': 'cpu',
        'channels': channels,
        'num_kernels': num_kernels,
        'tree_depth': tree_depth,
        'receptive_field_size': receptive_field_size,
        'implementation': 'python',
        'connections': 'random',
        'stride': stride,
        'padding': padding
    }
    # in_dim can be an integer or a tuple of integers. be m either the int itself or the min of the tuple
    if receptive_field_size > (min(in_dim) if isinstance(in_dim, tuple) else in_dim):
        with pytest.raises(AssertionError):
            LogicCNNLayer(**params)
        pytest.skip("Receptive field size should be smaller than input dimension")
    if stride > receptive_field_size:
        with pytest.raises(AssertionError):
            LogicCNNLayer(**params)
        pytest.skip("Stride should be smaller than receptive field size")
    return LogicCNNLayer(**params)


@pytest.mark.parametrize("in_dim", [2, 7, (18, 14)])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("num_kernels", [1, 5])
@pytest.mark.parametrize("tree_depth", [1, 3])
@pytest.mark.parametrize("receptive_field_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("side", [0, 1], ids=['left', 'right'])
class TestIndeces:
    """
    A layer's indices are of the shape [level_N, level_N-1, ..., level_0], where N is the tree depth.
    Each level is of shape (left_indices, right_indices), defining the inputs for the binary logic gates.
    """
    def test_first_tree_level_shape(self, layer, side):
        """
        The first tree level defines which entries within the receptive field are considered.
        It should be of shape (num_kernels, num_positions, 2**tree_depth, 3) [3 because of (w, h, c) notation]
        Also, 
        """
        vertical_positions = int((layer.in_dim[0] + 2*layer.padding - layer.receptive_field_size) / layer.stride) + 1
        horizontal_positions = int((layer.in_dim[1] + 2*layer.padding - layer.receptive_field_size) / layer.stride) + 1
        num_positions = horizontal_positions * vertical_positions
        indices = layer.indices[0][side]
        assert indices.shape == (layer.num_kernels, num_positions, 2**layer.tree_depth, 3)


    def test_other_tree_levels_shape(self, layer, side):
        """
        Since the convolution is implemented as a binary tree, all following levels 
        should have 2**i gates, where i is the level (in reverse order).
        """
        for level in range(1, layer.tree_depth):
            indices = layer.indices[level][side]
            expected_gates = 2**(layer.tree_depth - level)
            assert indices.shape == (expected_gates,)
    

    def test_first_tree_level_range(self, layer, side):
        """
        width, height and channel indices should be within specified input dimensions
        """
        indices = layer.indices[0][side]
        assert torch.all(indices[..., 0] < layer.in_dim[0])
        assert torch.all(indices[..., 1] < layer.in_dim[1])
        assert torch.all(indices[..., 2] < layer.channels)


    def test_other_tree_levels_range(self, layer, side):
        """
        Each following level should have indices within the range of the previous level
        """
        for level in range(1, layer.tree_depth):
            indices = layer.indices[level][side]
            n_gates_prev = 2**(layer.tree_depth - level + 1)
            assert torch.all(indices < n_gates_prev)


def test_and_model():
    """
    AND is the 1-st gate
     - set the weights to 0, except for the 1-st element (set to some high value)
     - test the 4 possible inputs
    """
    layer = LogicCNNLayer(
        in_dim=2,
        device='cpu',
        channels=1,
        num_kernels=1,
        tree_depth=1,
        receptive_field_size=2,
        implementation='python',
        connections='random',
        stride=1,
        padding=0
    )

    # Set indices to select pairs of positions row-wise
    layer.indices = [
        (
            torch.tensor([[[[0, 0, 0], [1, 0, 0]]]]),
            torch.tensor([[[[0, 1, 0], [1, 1, 0]]]])
        ),
        (
            torch.tensor([0]),
            torch.tensor([1])
        )
    ]

    # Set weights to select AND operation
    with torch.no_grad():
        and_weights = torch.zeros(1, 16)
        and_weights[0, 1] = 100.0  # Large value so softmax will make it close to 1
        layer.tree_weights[0][0].data = and_weights
        layer.tree_weights[0][1].data = and_weights
        layer.tree_weights[1][0].data = and_weights

    test_cases = [
        ((0,0), 0),
        ((0,1), 0),
        ((1,0), 0),
        ((1,1), 1)
    ]

    # only all 1s should produce 1
    test_cases = [
        ([[[[0, 0], [0, 0]]]], 0),
        ([[[[0, 0], [0, 1]]]], 0),
        ([[[[0, 0], [1, 1]]]], 0),
        ([[[[0, 1], [1, 1]]]], 0),
        ([[[[1, 1], [1, 1]]]], 1),
    ]
    
    for x, expected in test_cases:
        x = torch.tensor(x, dtype=torch.float32)
        output = layer(x)
        assert torch.isclose(output, torch.tensor([[[[expected]]]], dtype=torch.float32))
