import pytest
import torch
import os
import numpy as np
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet, LogicCNNLayer


llkw = {
    'connections': 'unique',
    'implementation': 'python',
    'device': 'cpu'
}

@pytest.mark.parametrize("tree_depth", [1,2,3,4,5])
def test_index_dimensions(tree_depth):
    """
    A layer's indices are of the shape [level_N, level_N-1, ..., level_0], where N is the tree depth.
    Each level is of shape (left_indices, right_indices), defining the inputs for the binary logic gates.
    As the levels form a binary tree, level_i should define i**N gates.
    level_N defines which entries within the receptive field are considered.
    It should be of shape (num_kernels, num_positions, 2**N, 3) [3 because of (w, h, c) notation]
    All following levels should be of shape (2**i,) with the last one of shape (1,)
    """
    m = LogicCNNLayer(
        in_dim=2,
        device='cpu',
        channels=1,
        num_kernels=1,
        tree_depth=tree_depth,
        receptive_field_size=2,
        implementation='python',
        connections='random',
        stride=1,
        padding=0
    )
    assert m.indices[-1][0].shape == (1,)
    assert m.indices[-1][1].shape == (1,)
    for level in range(tree_depth, 0):
        assert m.indices[level][0].shape == (level**2,)
