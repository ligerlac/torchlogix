"""Integration tests for the alkaid ALIR tracer plugin (torchlogix._alkaid_plugin).

Skipped entirely when the optional `alkaid` extra isn't installed:
    pip install torchlogix[alkaid]

Model/input fixtures (conv2d_model, sample_input_2d, etc.) live in conftest.py,
shared with test_export_mode.py.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

alkaid = pytest.importorskip("alkaid")

from alkaid.converter import trace_model
from alkaid.trace import FVArrayInput, trace

from torchlogix.utils import set_export_mode
from torchlogix.layers import LogicConv2d, LogicDense


def _assert_matches_eval(model, x):
    """Trace `model` through the torchlogix alkaid plugin and compare the
    resulting combinational circuit's prediction against plain eval-mode
    output on the same (boolean) input."""
    # Baseline: plain eval-mode forward, BEFORE switching to export mode
    # (accepts float bool-valued tensors, same convention as test_export_mode.py).
    expected = model(x.float()).detach().numpy().reshape(x.shape[0], -1)

    set_export_mode(model)
    input_shape = tuple(x.shape[1:])

    inp = FVArrayInput((1, *input_shape)).quantize(0, 1, 0)
    inp2, out = trace_model(model, inputs=inp, framework="torchlogix")
    comb = trace(inp2, out)

    actual = comb.predict(x.numpy())

    assert np.array_equal(expected, actual), (
        "alkaid comb.predict() diverges from eval-mode output"
    )


@pytest.mark.parametrize(
    "model_fixture, input_fixture",
    [
        ("logic_dense_model", "sample_input_1d"),
        ("conv2d_model_wo_group_sum", "sample_input_2d"),
        ("conv3d_model_wo_group_sum", "sample_input_3d"),
        ("conv2d_model", "sample_input_2d"),
        ("conv3d_model", "sample_input_3d"),
        ("single_3d_conv_model", "sample_input_3d"),
    ],
)
def test_plugin_matches_eval_mode(model_fixture, input_fixture, request):
    model = request.getfixturevalue(model_fixture)
    x = request.getfixturevalue(input_fixture)
    _assert_matches_eval(model, x)


@pytest.mark.parametrize("tree_depth", [1, 2, 3])
def test_logic_conv2d_tree_depths(sample_input_2d, tree_depth):
    model = LogicConv2d(
        in_dim=8, channels=3, num_kernels=6,
        receptive_field_size=3, tree_depth=tree_depth,
    )
    model.eval()
    _assert_matches_eval(model, sample_input_2d)


@pytest.mark.xfail(
    reason=(
        "Regression case for an alkaid core bug (see ZERO_CONSTANT_BUG.md in "
        "the alkaid repo, reported upstream): AffineInterval.qint mis-derives "
        "the bit-width of zero-valued constants (get_lsb_loc(0.0) == 127). "
        "Any neuron whose LUT happens to be CONST_FALSE (id 0, ~1/16 with "
        "random init) creates one; deeper/wider stacks make hitting one "
        "increasingly likely, and comb.predict() then diverges from eval-mode "
        "output. This exact seed/shape is a known deterministic repro against "
        "vanilla (unpatched) alkaid; passes once the local zero-constant fix "
        "is applied."
    ),
    strict=False,
)
def test_deep_stack_of_dense_layers_zero_constant_regression():
    torch.manual_seed(13)
    model = nn.Sequential(*[
        LogicDense(16, 16, parametrization="raw", parametrization_kwargs={"weight_init": "random"})
        for _ in range(3)
    ])
    model.eval()
    x = torch.randint(0, 2, (32, 16)).bool()
    _assert_matches_eval(model, x)
