import math
import pytest
import torch
from torchlogix.layers import LogicDense
from torchlogix.connections import LearnableDenseConnections, FixedConvConnections, FixedConvTransposeConnections
from torch.nn.functional import softmax as softmax_torch


@pytest.mark.parametrize("parametrization", ["raw", "warp", "light"])
@pytest.mark.parametrize("num_candidates", [-1, 1, 2, 3])
@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_learnable_connections(parametrization, num_candidates, lut_rank):
    """Test that connections can be trained."""
    parametrization_kwargs = {
        "weight_init": "residual",
        "residual_probability": 0.9
    }
    if lut_rank > 2 and parametrization == "raw":
        pytest.skip("Raw parametrization currently only supports lut_rank=2 ")
    connections_kwargs = {"init_method": "random-unique", "num_candidates": num_candidates}
    in_dim = 100
    out_dim = 100
    layer = LogicDense(in_dim=in_dim, 
                       out_dim=out_dim, 
                       lut_rank=lut_rank, 
                       connections="learnable",
                       connections_kwargs=connections_kwargs, 
                       device="cpu",
                       parametrization=parametrization,
                       parametrization_kwargs=parametrization_kwargs)
    if num_candidates == -1:
        assert layer.connections.indices.shape[0] == layer.in_dim
    else:
        assert layer.connections.indices.shape[0] == num_candidates
    assert layer.connections.indices.shape[1] == layer.lut_rank
    assert layer.connections.indices.shape[2] == layer.out_dim
    assert layer.connections.indices.shape == layer.connections.weights.shape
    X = torch.rand((5, in_dim), requires_grad=True)
    layer.training = True
    y = layer(X)
    loss = y.sum()
    loss.backward()
    assert all(torch.norm(p.grad) > 0 for p in layer.parameters())


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_learnable_gradients(lut_rank):
    """Test that gradients flow through learnable connections."""
    connections_kwargs = {"init_method": "random", "num_candidates": -1}
    in_dim = 100
    out_dim = 100
    layer = LearnableDenseConnections(
        in_dim=in_dim, 
        out_dim=out_dim, 
        lut_rank=lut_rank, 
        device="cpu",
        temperature=1,
        **connections_kwargs
        )
    parameters = [p for p in layer.parameters()]
    X = torch.rand((100, in_dim), requires_grad=True)
    y = layer(X)
    y.retain_grad()
    loss = y.sum()
    loss.backward()
    # DWN computation forward
    weights = parameters[0].flatten(start_dim=-2)
    mapping = weights.argmax(dim=0)
    output = X[:, mapping]
    assert torch.allclose(y.flatten(start_dim=-2), output)
    # DWN computation backward
    output_grad = y.grad.flatten(start_dim=-2)
    weights_grad = ((2*X-1).T @ output_grad)
    assert torch.allclose(parameters[0].grad.flatten(start_dim=-2), weights_grad, atol=1e-3, rtol=1e-3)
    input_grad = output_grad @ softmax_torch(weights, dim=0).T
    assert torch.allclose(X.grad, input_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("channel_group_size", [None, 1, 2])
def test_fixed_conv_connections(channel_group_size):
    """
    indices: (lut_rank, num_kernels, kernel_position, sample_size, 3)
        where the last dim is (h, w, c)
        for each tree level
    """
    num_kernels = 3

    conn = FixedConvConnections(
        in_dim=28, channels=3, num_kernels=num_kernels, tree_depth=3, receptive_field_size=3, channel_group_size=channel_group_size
    )

    # only the first matters (field of view)
    fow_indices = conn.indices[0]

    for kernel_idx in range(num_kernels):
        considered_channels = fow_indices[:, kernel_idx, :, :, 2].unique()
        if channel_group_size is not None:
            assert len(considered_channels) <= channel_group_size, (
                "channel_group_size must be smaller than the number of channels"
            )

@pytest.mark.parametrize("stride,output_padding", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 2)])
def test_fixed_conv_transpose_connections_output_shape(stride, output_padding):
    """Index tensors must produce the correct transposed-conv output spatial size."""
    in_h, in_w = 6, 6
    kH = 3
    padding = 0
    channels = 2
    num_kernels = 4
    tree_depth = 2
    batch = 2

    conn = FixedConvTransposeConnections(
        in_dim=(in_h, in_w),
        channels=channels,
        num_kernels=num_kernels,
        tree_depth=tree_depth,
        receptive_field_size=kH,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        conv_dimension=2,
    )

    expected_out = (in_h - 1) * stride - 2 * padding + kH + output_padding

    # Level-0 indices encode spatial positions
    fov = conn.indices[0]           # (lut_rank, num_kernels, num_positions, sample_size, 3)
    num_positions = fov.shape[2]
    assert num_positions == expected_out ** 2, (
        f"Expected {expected_out**2} output positions, got {num_positions}"
    )

    # Verify forward produces the right gathered shape at level 0.
    x = torch.rand(batch, channels, in_h, in_w)
    out = conn(x, tree_level=0)
    assert out.shape[3] == expected_out ** 2


# ---------------------------------------------------------------------------
# Convolutional connection indices
#
# A layer's indices are [level_0, level_1, ..., level_N-1]. Level 0 selects
# entries within the receptive field and has shape
# (lut_rank, num_kernels, num_positions, 2**(tree_depth-1), ndim + 1), where the
# last axis is (w, h, c) in 2D and (w, h, d, c) in 3D. Every later level picks
# among the previous level's gates and has shape (lut_rank, 2**(depth-level-1)).
#
# These assertions used to live in a TestIndeces / TestIndices class in
# test_clgn.py and test_clgn_3d.py as a nine-axis cartesian product: ~5,900
# cases, half of which skipped because the product generated combinations that
# are invalid by construction. The table below instead lists valid
# configurations explicitly.
#
# To cover a new case, add one pytest.param line.
# ---------------------------------------------------------------------------

CONV_CONFIGS = [
    # --- 2D ---
    pytest.param(2, dict(in_dim=2, rfs=2, stride=1, padding=0, tree_depth=1,
                         channels=1, num_kernels=1), id="2d-minimal"),
    pytest.param(2, dict(in_dim=7, rfs=3, stride=1, padding=0, tree_depth=3,
                         channels=2, num_kernels=5), id="2d-square"),
    pytest.param(2, dict(in_dim=7, rfs=3, stride=1, padding=1, tree_depth=2,
                         channels=2, num_kernels=3), id="2d-padding-1"),
    pytest.param(2, dict(in_dim=7, rfs=3, stride=2, padding=2, tree_depth=2,
                         channels=1, num_kernels=2), id="2d-padding-2"),
    pytest.param(2, dict(in_dim=7, rfs=3, stride=3, padding=0, tree_depth=2,
                         channels=2, num_kernels=5), id="2d-stride-3"),
    pytest.param(2, dict(in_dim=(18, 14), rfs=3, stride=3, padding=0, tree_depth=3,
                         channels=2, num_kernels=5), id="2d-rectangular"),
    pytest.param(2, dict(in_dim=(9, 7), rfs=(3, 2), stride=1, padding=0, tree_depth=2,
                         channels=2, num_kernels=3), id="2d-anisotropic-rfs"),
    pytest.param(2, dict(in_dim=8, rfs=3, stride=1, padding=0, tree_depth=4,
                         channels=2, num_kernels=2), id="2d-deep-tree"),
    pytest.param(2, dict(in_dim=5, rfs=2, stride=2, padding=0, tree_depth=1,
                         channels=1, num_kernels=1), id="2d-single-channel"),
    # --- 3D ---
    pytest.param(3, dict(in_dim=2, rfs=2, stride=1, padding=0, tree_depth=2,
                         channels=1, num_kernels=1), id="3d-minimal"),
    pytest.param(3, dict(in_dim=7, rfs=3, stride=1, padding=0, tree_depth=4,
                         channels=2, num_kernels=5), id="3d-cubic"),
    pytest.param(3, dict(in_dim=6, rfs=3, stride=1, padding=1, tree_depth=2,
                         channels=2, num_kernels=3), id="3d-padding-1"),
    pytest.param(3, dict(in_dim=7, rfs=3, stride=3, padding=0, tree_depth=2,
                         channels=2, num_kernels=3), id="3d-stride-3"),
    pytest.param(3, dict(in_dim=(18, 14, 6), rfs=3, stride=3, padding=0, tree_depth=2,
                         channels=2, num_kernels=5), id="3d-rectangular"),
    pytest.param(3, dict(in_dim=(9, 7, 5), rfs=(3, 2, 2), stride=1, padding=0, tree_depth=2,
                         channels=2, num_kernels=3), id="3d-anisotropic-rfs"),
]

INIT_METHODS = ["random", "random-unique"]
SIDES = [pytest.param(0, id="left"), pytest.param(1, id="right")]


def make_conv_connections(ndim, config, init_method="random"):
    """Build FixedConvConnections straight from a CONV_CONFIGS entry.

    Built directly rather than through a layer: these are connection
    properties, and going via a layer would also make the indices depend on
    weight initialization, since a layer builds its weights first and thereby
    shifts the RNG state.
    """
    return FixedConvConnections(
        in_dim=config["in_dim"],
        channels=config["channels"],
        num_kernels=config["num_kernels"],
        tree_depth=config["tree_depth"],
        receptive_field_size=config["rfs"],
        stride=config["stride"],
        padding=config["padding"],
        conv_dimension=ndim,
        init_method=init_method,
    )


def expected_num_positions(conn):
    """Number of sliding-window positions the connections should cover."""
    return math.prod(
        (dim + 2 * conn.padding - rfs) // conn.stride + 1
        for dim, rfs in zip(conn.in_dim, conn.receptive_field_size)
    )


@pytest.mark.parametrize("ndim, config", CONV_CONFIGS)
@pytest.mark.parametrize("init_method", INIT_METHODS)
@pytest.mark.parametrize("side", SIDES)
def test_first_level_index_shape(ndim, config, init_method, side):
    """Level 0 selects receptive-field entries for every sliding position."""
    conn = make_conv_connections(ndim, config, init_method)
    indices = conn.indices[0][side]
    assert indices.shape == (
        conn.num_kernels,
        expected_num_positions(conn),
        2 ** (conn.tree_depth - 1),
        ndim + 1,
    )


@pytest.mark.parametrize("ndim, config", CONV_CONFIGS)
@pytest.mark.parametrize("init_method", INIT_METHODS)
@pytest.mark.parametrize("side", SIDES)
def test_other_levels_index_shape(ndim, config, init_method, side):
    """Each level above 0 halves the number of gates, as a binary tree."""
    conn = make_conv_connections(ndim, config, init_method)
    for level in range(1, conn.tree_depth):
        indices = conn.indices[level][side]
        assert indices.shape == (2 ** (conn.tree_depth - level - 1),)


@pytest.mark.parametrize("ndim, config", CONV_CONFIGS)
@pytest.mark.parametrize("init_method", INIT_METHODS)
@pytest.mark.parametrize("side", SIDES)
def test_first_level_indices_within_input_bounds(ndim, config, init_method, side):
    """Spatial indices must stay inside the padded input; channels inside channels.

    The bound is in_dim + 2 * padding, not in_dim: the layer pads before
    gathering, so level-0 indices address the padded tensor. The old version of
    this assertion compared against in_dim alone and was simply never exercised
    with padding, since padding was pinned to 0.
    """
    conn = make_conv_connections(ndim, config, init_method)
    indices = conn.indices[0][side]
    for axis, dim in enumerate(conn.in_dim):
        padded = dim + 2 * conn.padding
        assert torch.all(indices[..., axis] < padded), f"axis {axis} index out of bounds"
    assert torch.all(indices[..., ndim] < conn.channels), "channel index out of bounds"
    assert torch.all(indices >= 0), "negative index"


@pytest.mark.parametrize("ndim, config", CONV_CONFIGS)
@pytest.mark.parametrize("init_method", INIT_METHODS)
@pytest.mark.parametrize("side", SIDES)
def test_other_levels_indices_within_previous_level(ndim, config, init_method, side):
    """Every level may only reference gates that the level below actually has."""
    conn = make_conv_connections(ndim, config, init_method)
    for level in range(1, conn.tree_depth):
        indices = conn.indices[level][side]
        n_gates_prev = 2 ** (conn.tree_depth - level + 1)
        assert torch.all(indices < n_gates_prev)
        assert torch.all(indices >= 0)


@pytest.mark.parametrize("ndim, config", CONV_CONFIGS)
def test_random_unique_first_level_pairs_are_unique(ndim, config):
    """random-unique must not feed the same input pair to two gates.

    The pair is unordered: (a, b) and (b, a) are the same wiring.
    """
    conn = make_conv_connections(ndim, config, init_method="random-unique")
    left, right = conn.indices[0][0], conn.indices[0][1]

    for kernel_idx in range(left.shape[0]):
        for pos_idx in range(left.shape[1]):
            left_pos = left[kernel_idx, pos_idx]
            right_pos = right[kernel_idx, pos_idx]
            pairs = set()
            for i in range(left_pos.shape[0]):
                a, b = tuple(left_pos[i].tolist()), tuple(right_pos[i].tolist())
                pairs.add((a, b) if a < b else (b, a))
            assert len(pairs) == left_pos.shape[0], (
                f"duplicate input pair at kernel {kernel_idx}, position {pos_idx}"
            )


# ---------------------------------------------------------------------------
# Invalid configurations
#
# These used to be asserted invisibly inside a fixture, via pytest.skip and a
# pytest.raises that only ran for combinations the cartesian product happened
# to generate. Stating them directly is both clearer and cheaper.
# ---------------------------------------------------------------------------

def test_rejects_receptive_field_larger_than_input():
    with pytest.raises(AssertionError, match="must fit within input dimensions"):
        make_conv_connections(2, dict(in_dim=2, rfs=3, stride=1, padding=0,
                                      tree_depth=1, channels=1, num_kernels=1))


def test_rejects_stride_larger_than_receptive_field():
    from torchlogix.layers import LogicConv2d
    with pytest.raises(AssertionError, match="Stride"):
        LogicConv2d(in_dim=7, channels=1, num_kernels=1, tree_depth=1,
                    receptive_field_size=2, stride=3, device="cpu")


def test_rejects_unknown_conv_dimension():
    with pytest.raises(AssertionError, match="conv_dimension must be 2 or 3"):
        make_conv_connections(4, dict(in_dim=4, rfs=2, stride=1, padding=0,
                                      tree_depth=1, channels=1, num_kernels=1))


def test_rejects_unknown_init_method():
    with pytest.raises(ValueError, match="Unknown connections type"):
        make_conv_connections(2, dict(in_dim=4, rfs=2, stride=1, padding=0,
                                      tree_depth=1, channels=1, num_kernels=1),
                              init_method="nonsense")


def test_rejects_unique_pairs_when_kernel_volume_too_small():
    """random-unique needs enough distinct pairs to wire a tree of that depth."""
    with pytest.raises(ValueError, match="Not enough unique combinations"):
        make_conv_connections(2, dict(in_dim=4, rfs=2, stride=1, padding=0,
                                      tree_depth=5, channels=1, num_kernels=1),
                              init_method="random-unique")

@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_dense_unique_connections_cover_all_inputs_evenly(lut_rank):
    """random-unique dense wiring must use every input, and use them evenly.

    Each output column draws lut_rank distinct inputs, every input is used at
    least once, and the usage counts stay within one of each other.
    """
    layer = LogicDense(in_dim=400, out_dim=400, lut_rank=lut_rank,
                       connections="fixed", device="cpu", parametrization="warp",
                       connections_kwargs={"init_method": "random-unique"})
    indices = layer.connections.indices

    for col in range(indices.shape[1]):
        assert len(torch.unique(indices[..., col])) == lut_rank

    unique, counts = torch.unique(indices, return_counts=True)
    assert counts.float().std().item() < 1, "input usage is not balanced"
    assert len(unique) == layer.in_dim, "not every input is used"
