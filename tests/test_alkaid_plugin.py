"""Integration tests for the alkaid ALIR tracer plugin (torchlogix._alkaid_plugin).

Skipped entirely when the optional `alkaid` extra isn't installed:
    pip install torchlogix[alkaid]

Model definitions live in tests/models.py, shared with the rest of the suite.
"""
import numpy as np
import pytest
import torch

alkaid = pytest.importorskip("alkaid")

from alkaid.converter import trace_model
from alkaid.trace import FVArrayInput, trace

from torchlogix.utils import set_export_mode

from models import (
    TORCHLOGIX_MODELS,
    in_place_const_mutation_model,
    random_bool_input,
)


@pytest.mark.parametrize("model_fn", TORCHLOGIX_MODELS)
def test_plugin_matches_eval_mode(model_fn):
    model = model_fn()
    model.eval()
    x = random_bool_input(model, batch_size=4, seed=0)

    set_export_mode(model)
    expected = model(x).detach().numpy().reshape(x.shape[0], -1)

    input_shape = tuple(x.shape[1:])
    inp = FVArrayInput((1, *input_shape)).quantize(0, 1, 0)
    inp2, out = trace_model(model, inputs=inp, framework="logic")
    comb = trace(inp2, out)

    actual = comb.predict(x.numpy())

    assert np.array_equal(expected, actual), (
        "alkaid comb.predict() diverges from eval-mode output"
    )


def test_alkaid_rejects_inplace_constant_mutation():
    model = in_place_const_mutation_model()
    x = random_bool_input(model, batch_size=4, seed=0)
    inp = FVArrayInput((1, *x.shape[1:])).quantize(0, 1, 0)
    with pytest.raises(NotImplementedError, match="unsupported constant-tensor mutation"):
        trace_model(model, inputs=inp, framework="logic")
