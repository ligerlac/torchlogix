"""Integration tests for the alkaid ALIR tracer plugin (torchlogix._alkaid_plugin).

Skipped entirely when the optional `alkaid` extra isn't installed:
    pip install torchlogix[alkaid]

Model/input fixtures (conv2d_model, sample_input_2d, etc.) live in conftest.py,
shared with test_export_mode.py.
"""
import numpy as np
import pytest

alkaid = pytest.importorskip("alkaid")

from alkaid.converter import trace_model
from alkaid.trace import FVArrayInput, trace

from torchlogix.utils import set_export_mode


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

    expected = model(x.float()).detach().numpy().reshape(x.shape[0], -1)
    set_export_mode(model)

    input_shape = tuple(x.shape[1:])
    inp = FVArrayInput((1, *input_shape)).quantize(0, 1, 0)
    inp2, out = trace_model(model, inputs=inp, framework="logic")
    comb = trace(inp2, out)

    actual = comb.predict(x.numpy())

    assert np.array_equal(expected, actual), (
        "alkaid comb.predict() diverges from eval-mode output"
    )
