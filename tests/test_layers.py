"""Layer properties that hold by formula, merged across layer families.

Everything here asserts something computable from the layer's configuration -
an output shape, a gradient reaching every parameter, a regularization loss
going to zero - so one parametrized test can cover the dense, 2D and 3D
families at once. Adding a family to a list below gives it the whole set.

Behavioural tests, where the expected value is a hand-computed truth table
tied to one spatial arrangement, deliberately stay unmerged in
test_gate_semantics.py.
"""
import numpy as np
import pytest
import torch

from torchlogix.utils import set_export_mode
from torchlogix.layers import (
    GroupSum,
    LogicConv2d,
    LogicConv3d,
    LogicConvTranspose2d,
    LogicConvTranspose3d,
    LogicDense,
)

from helpers import (
    assert_compiled_gradients_match_eager,
    assert_finite_difference_matches_autograd,
)

CONNECTIONS_KWARGS = {"init_method": "random-unique"}


# ---------------------------------------------------------------------------
# Regularization and weight rescaling
#
# These assertions are identical for dense and conv layers; only building the
# layer and writing its weights differs, which is what the builders absorb.
# (The dense and conv copies of these tests were previously separate, in
# test_lgn.py and test_clgn.py.)
# ---------------------------------------------------------------------------

def _dense_warp_layer():
    return LogicDense(in_dim=2, out_dim=1, connections="fixed", device="cpu",
                      parametrization="warp", connections_kwargs=CONNECTIONS_KWARGS)


def _conv_warp_layer(ndim):
    cls = LogicConv2d if ndim == 2 else LogicConv3d
    in_dim = (3, 4) if ndim == 2 else (3, 4, 4)
    return cls(in_dim=in_dim, parametrization="warp", device="cpu", channels=1,
               num_kernels=1, tree_depth=1, receptive_field_size=3 if ndim == 2 else 2,
               connections_kwargs=CONNECTIONS_KWARGS, stride=1, padding=0, lut_rank=2)


WARP_LAYERS = [
    pytest.param(_dense_warp_layer, id="dense"),
    pytest.param(lambda: _conv_warp_layer(2), id="conv2d"),
    pytest.param(lambda: _conv_warp_layer(3), id="conv3d"),
]


def _set_all_weights(layer, values):
    """Broadcast one Walsh-coefficient pattern into every parameter.

    Parameter shapes differ by family - (1, 4) for dense, (1, 1, 4) for conv -
    so the pattern is expanded rather than assigned as a literal.
    """
    pattern = torch.tensor(values, dtype=torch.float32)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(pattern.expand_as(p))


@pytest.mark.parametrize("make_layer", WARP_LAYERS)
@pytest.mark.parametrize("method", ["abs_sum", "L2"])
def test_regularization_loss_is_zero_for_balanced_weights(make_layer, method):
    """Balanced Walsh coefficients carry no regularization penalty."""
    layer = make_layer()
    _set_all_weights(layer, [0.5, 0.5, 0.5, -0.5])
    assert np.isclose(layer.get_regularization_loss(method).item(), 0.0)


@pytest.mark.parametrize("make_layer", WARP_LAYERS)
@pytest.mark.parametrize("method", ["abs_sum", "L2"])
def test_regularization_loss_is_positive_for_unbalanced_weights(make_layer, method):
    layer = make_layer()
    _set_all_weights(layer, [0.5, 0.5, 0.5, -0.5])
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(1.0)
    assert layer.get_regularization_loss(method).item() > 0.0


@pytest.mark.parametrize("make_layer", WARP_LAYERS)
@pytest.mark.parametrize("method", ["abs_sum", "L2"])
def test_rescale_weights_drives_regularization_loss_to_zero(make_layer, method):
    layer = make_layer()
    _set_all_weights(layer, [0.5, 0.5, 0.5, 1.0])
    assert layer.get_regularization_loss(method).item() > 0.0

    layer.rescale_weights(method)
    assert np.isclose(layer.get_regularization_loss(method).item(), 0.0)


# ---------------------------------------------------------------------------
# Dense layer specifics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("connections", ["fixed", "learnable"])
@pytest.mark.parametrize("parametrization", ["raw", "warp", "light"])
def test_dense_layer_broadcasts_leading_dims(connections, parametrization):
    """Dense layers use only the last dim as features and broadcast the rest.

    Reshaping to a flat batch must give the same result as the nested shape.
    """
    layer = LogicDense(in_dim=8, out_dim=8, connections=connections,
                       parametrization=parametrization)
    x = torch.rand((2, 3, 4, 8))
    out = layer(x)
    out_flat = layer(x.view(-1, 8)).view(2, 3, 4, 8)
    assert torch.allclose(out, out_flat)


def test_trivial_dense_layer_wires_its_only_connection():
    """2 inputs, 1 output leaves exactly one possible connection: 0 to 1.

    A second output would need a second distinct pair, which does not exist.
    """
    layer = LogicDense(in_dim=2, out_dim=1, connections="fixed", device="cpu",
                       connections_kwargs=CONNECTIONS_KWARGS)
    indices = layer.connections.indices
    assert (torch.allclose(indices, torch.tensor(((0,), (1,))))
            or torch.allclose(indices, torch.tensor(((1,), (0,)))))
    assert layer.weight.shape == (1, 16)

    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=2, connections="fixed", device="cpu",
                   connections_kwargs=CONNECTIONS_KWARGS)


@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_trivial_dense_layer_uses_every_input_once(lut_rank):
    """With in_dim == lut_rank the single gate must consume all inputs.

    init_method is passed explicitly rather than mutating a shared kwargs dict,
    which is what the previous version of this test did - leaking "random" into
    every test that ran after it.
    """
    layer = LogicDense(in_dim=lut_rank, out_dim=1, lut_rank=lut_rank,
                       connections="fixed", device="cpu", parametrization="warp",
                       connections_kwargs={"init_method": "random"})
    assert layer.connections.indices.shape == (lut_rank, 1)
    assert set(layer.connections.indices[:, 0].tolist()) == set(range(lut_rank))
    assert layer.weight.shape == (1, 2 ** lut_rank)


@pytest.mark.parametrize("lut_rank", [4, 6])
def test_dense_rejects_in_dim_below_lut_rank(lut_rank):
    """A gate cannot draw more distinct inputs than the layer has.

    Uses warp: the raw parametrization only supports lut_rank=2 and would raise
    for a different reason.
    """
    with pytest.raises(AssertionError):
        LogicDense(in_dim=2, out_dim=1, lut_rank=lut_rank, connections="fixed",
                   device="cpu", parametrization="warp",
                   connections_kwargs=CONNECTIONS_KWARGS)


# ---------------------------------------------------------------------------
# Transposed convolution
#
# The 2D and 3D versions of these were separate, near-identical tests in
# test_clgn.py and test_clgn_3d.py. Their assertions are formulas, so they
# merge cleanly: each family only contributes its layer classes and a base
# input size.
# ---------------------------------------------------------------------------

TRANSPOSE_FAMILIES = [
    pytest.param(2, LogicConv2d, LogicConvTranspose2d, 6, id="2d"),
    pytest.param(3, LogicConv3d, LogicConvTranspose3d, 4, id="3d"),
]

# stride/output_padding pairs are all valid by construction (output_padding
# must be < stride), so nothing here skips.
STRIDE_OUTPUT_PADDING = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 2)]


@pytest.mark.parametrize("ndim, conv_cls, transpose_cls, in_dim", TRANSPOSE_FAMILIES)
@pytest.mark.parametrize("stride, output_padding", STRIDE_OUTPUT_PADDING)
@pytest.mark.parametrize("padding", [0, 1])
def test_conv_transpose_output_shape(ndim, conv_cls, transpose_cls, in_dim,
                                     stride, output_padding, padding):
    """out = (in - 1) * stride - 2 * padding + receptive_field + output_padding."""
    rfs, channels, num_kernels, batch = 3, 2, 4, 2
    layer = transpose_cls(
        in_dim=tuple([in_dim] * ndim), channels=channels, num_kernels=num_kernels,
        tree_depth=2, receptive_field_size=rfs, stride=stride, padding=padding,
        output_padding=output_padding, device="cpu",
    )
    out = layer(torch.rand(batch, channels, *([in_dim] * ndim)))

    expected = (in_dim - 1) * stride - 2 * padding + rfs + output_padding
    assert out.shape == (batch, num_kernels, *([expected] * ndim))


@pytest.mark.parametrize("ndim, conv_cls, transpose_cls, in_dim", TRANSPOSE_FAMILIES)
def test_conv_transpose_gradients_reach_every_tree_level(ndim, conv_cls, transpose_cls, in_dim):
    layer = transpose_cls(
        in_dim=tuple([in_dim] * ndim), channels=2, num_kernels=4, tree_depth=2,
        receptive_field_size=3, stride=2, padding=0, output_padding=0,
        parametrization="warp", device="cpu",
    )
    layer.train()
    layer(torch.rand(2, 2, *([in_dim] * ndim))).sum().backward()

    for i, w in enumerate(layer.tree_weights):
        assert w.grad is not None, f"tree_weights[{i}] has no gradient"
        assert w.grad.abs().sum() > 0, f"tree_weights[{i}] gradient is all zero"


@pytest.mark.parametrize("ndim, conv_cls, transpose_cls, in_dim", TRANSPOSE_FAMILIES)
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv_transpose_inverts_conv_spatial_dims(ndim, conv_cls, transpose_cls,
                                                  in_dim, stride, padding):
    """A transpose conv must undo the spatial mapping of the matching conv.

    output_padding is chosen as the remainder the forward conv discarded, which
    is exactly what makes the round trip land back on the original size.
    """
    rfs, channels, num_kernels, batch = 3, 2, 4, 2
    conv = conv_cls(
        in_dim=tuple([in_dim] * ndim), channels=channels, num_kernels=num_kernels,
        tree_depth=2, receptive_field_size=rfs, stride=stride, padding=padding,
        device="cpu",
    )
    conv_out = conv(torch.rand(batch, channels, *([in_dim] * ndim)))
    latent = conv_out.shape[2]

    transpose = transpose_cls(
        in_dim=tuple([latent] * ndim), channels=num_kernels, num_kernels=channels,
        tree_depth=2, receptive_field_size=rfs, stride=stride, padding=padding,
        output_padding=(in_dim + 2 * padding - rfs) % stride, device="cpu",
    )
    assert transpose(conv_out).shape == (batch, channels, *([in_dim] * ndim))


@pytest.mark.parametrize("ndim, conv_cls, transpose_cls, in_dim", TRANSPOSE_FAMILIES)
def test_conv_transpose_kernel_positions_describe_enlarged_output(
        ndim, conv_cls, transpose_cls, in_dim):
    """kernel_positions must describe the transposed output, not the conv one.

    The base class computes it with the forward-conv formula; without the
    override the export-mode LUT buffers come out the wrong size.
    """
    layer = transpose_cls(
        in_dim=tuple([in_dim] * ndim), channels=2, num_kernels=4, tree_depth=2,
        receptive_field_size=3, stride=2, padding=0, output_padding=0, device="cpu",
    )
    out = layer(torch.rand(1, 2, *([in_dim] * ndim)))
    assert layer.kernel_positions == list(out.shape[2:])
    assert layer.n_kernel_positions == int(np.prod(out.shape[2:]))


@pytest.mark.parametrize("ndim, conv_cls, transpose_cls, in_dim", TRANSPOSE_FAMILIES)
def test_conv_transpose_dilation_covers_every_spatial_axis(
        ndim, conv_cls, transpose_cls, in_dim):
    """Every spatial axis must be dilated - a 2D-shaped pad would skip depth."""
    stride = 2
    layer = transpose_cls(
        in_dim=tuple([in_dim] * ndim), channels=1, num_kernels=1, tree_depth=1,
        receptive_field_size=2, stride=stride, padding=1, output_padding=0, device="cpu",
    )
    x = torch.rand(1, 1, *([in_dim] * ndim))
    dilated = layer.connections._make_dilated_input(x)

    expected = (in_dim - 1) * stride + 1
    assert tuple(dilated.shape) == (1, 1, *([expected] * ndim))

    # Original samples land on the stride grid; the interleaved slots are zero.
    grid = (slice(None), slice(None)) + tuple(slice(None, None, stride) for _ in range(ndim))
    assert torch.equal(dilated[grid], x)
    off_grid = (slice(None), slice(None)) + tuple(1 for _ in range(ndim))
    assert dilated[off_grid].abs().sum() == 0


# ---------------------------------------------------------------------------
# Conv / transpose-conv autoencoders
# ---------------------------------------------------------------------------

# in_dim -> latent size under conv(rfs=3, stride=2), and back with
# output_padding=1. Both families genuinely round-trip.
AE_SIZES = {2: (8, 3), 3: (6, 2)}


def _make_ae(ndim, parametrization="raw"):
    conv_cls = LogicConv2d if ndim == 2 else LogicConv3d
    transpose_cls = LogicConvTranspose2d if ndim == 2 else LogicConvTranspose3d
    in_dim, latent = AE_SIZES[ndim]
    encoder = conv_cls(
        in_dim=in_dim, channels=2, num_kernels=4 if ndim == 2 else 3,
        receptive_field_size=3, tree_depth=2, stride=2, padding=0, device="cpu",
        parametrization=parametrization, parametrization_kwargs={"weight_init": "random"},
    )
    decoder = transpose_cls(
        in_dim=latent, channels=4 if ndim == 2 else 3, num_kernels=2,
        receptive_field_size=3, tree_depth=2, stride=2, padding=0, output_padding=1,
        device="cpu", parametrization=parametrization,
        parametrization_kwargs={"weight_init": "random"},
    )
    return torch.nn.Sequential(encoder, decoder)


@pytest.mark.parametrize("ndim", [2, 3])
def test_conv_ae_returns_to_input_resolution(ndim):
    model = _make_ae(ndim)
    in_dim, latent = AE_SIZES[ndim]
    x = torch.rand(3, 2, *([in_dim] * ndim))

    encoded = model[0](x)
    assert tuple(encoded.shape[2:]) == tuple([latent] * ndim)

    decoded = model(x)
    assert tuple(decoded.shape) == (3, 2, *([in_dim] * ndim)), \
        f"AE must return to the input resolution, got {tuple(decoded.shape)}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_conv_ae_trains_end_to_end(ndim):
    """A reconstruction loss must produce gradients in encoder and decoder alike."""
    torch.manual_seed(0)
    model = _make_ae(ndim, parametrization="warp")
    model.train()

    in_dim, _ = AE_SIZES[ndim]
    x = (torch.rand(2, 2, *([in_dim] * ndim)) > 0.5).float()
    torch.nn.functional.mse_loss(model(x), x).backward()

    for name, layer in (("encoder", model[0]), ("decoder", model[1])):
        for i, w in enumerate(layer.tree_weights):
            assert w.grad is not None, f"{name}.tree_weights[{i}] has no gradient"
            assert w.grad.abs().sum() > 0, f"{name}.tree_weights[{i}] gradient is all zero"


@pytest.mark.parametrize("ndim", [2, 3])
def test_conv_ae_is_deterministic_in_eval(ndim):
    """Eval mode must be free of sampling noise: repeated calls agree exactly."""
    model = _make_ae(ndim)
    model.eval()
    in_dim, _ = AE_SIZES[ndim]
    x = (torch.rand(2, 2, *([in_dim] * ndim)) > 0.5).float()
    assert torch.equal(model(x), model(x))


# ---------------------------------------------------------------------------
# Gradient correctness
#
# The tests above only check that a gradient is non-zero. These check that it
# is the *right* gradient, by comparing against central finite differences,
# and that torch.compile does not change it.
# ---------------------------------------------------------------------------

def _grad_check_layer(kind, parametrization, seed=0):
    """A small layer of each family, sized so finite differences stay cheap.

    Seeded explicitly: layer construction draws from the global RNG, so without
    this the weights - and therefore the gradient magnitudes these tests
    compare - depend on how many tests happened to run first.
    """
    torch.manual_seed(seed)
    kwargs = dict(device="cpu", parametrization=parametrization,
                  parametrization_kwargs={"weight_init": "random"})
    if kind == "dense":
        return LogicDense(in_dim=8, out_dim=4, connections="fixed", **kwargs)
    conv_cls = {"conv2d": LogicConv2d, "conv3d": LogicConv3d,
                "transpose2d": LogicConvTranspose2d, "transpose3d": LogicConvTranspose3d}[kind]
    ndim = 2 if kind.endswith("2d") else 3
    extra = {"output_padding": 0} if kind.startswith("transpose") else {}
    return conv_cls(in_dim=tuple([5] * ndim), channels=2, num_kernels=3,
                    receptive_field_size=2, tree_depth=2, stride=1, padding=0,
                    **extra, **kwargs)


GRAD_CHECK_KINDS = ["dense", "conv2d", "conv3d", "transpose2d", "transpose3d"]


def _layer_input(layer, kind, batch=2, seed=0):
    torch.manual_seed(seed)
    if kind == "dense":
        return (torch.rand(batch, layer.in_dim) > 0.5).float()
    return (torch.rand(batch, layer.channels, *layer.in_dim) > 0.5).float()


@pytest.mark.parametrize("kind", GRAD_CHECK_KINDS)
@pytest.mark.parametrize("parametrization", ["raw", "warp", "light"])
def test_layer_gradients_match_finite_differences(kind, parametrization):
    layer = _grad_check_layer(kind, parametrization)
    layer.train()
    x = _layer_input(layer, kind)
    torch.manual_seed(1)
    weights = torch.rand_like(layer(x))
    assert_finite_difference_matches_autograd(layer, x, weights)


@pytest.mark.slow
@pytest.mark.parametrize("kind", GRAD_CHECK_KINDS)
def test_layer_gradients_survive_torch_compile(kind):
    """Compiling a layer must not change its gradients.

    Marked slow: each case pays a one-off torch.compile cost of a few seconds.
    Deselect with -m "not slow".
    """
    layer = _grad_check_layer(kind, "raw")
    layer.train()
    x = _layer_input(layer, kind)
    torch.manual_seed(1)
    weights = torch.rand_like(layer(x))
    assert_compiled_gradients_match_eager(layer, x, weights)


# ---------------------------------------------------------------------------
# Eval mode vs export mode, at layer granularity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", GRAD_CHECK_KINDS)
def test_layer_eval_and_export_agree(kind):
    """Export mode must reproduce the thresholded eval-mode output exactly."""
    layer = _grad_check_layer(kind, "raw")
    layer.eval()
    x = _layer_input(layer, kind)

    soft = layer(x)
    set_export_mode(layer)
    hard = layer(x.bool())

    assert hard.dtype == torch.bool, f"export mode should return bool, got {hard.dtype}"
    assert torch.equal(hard, soft > 0.5), "export output differs from thresholded eval output"
