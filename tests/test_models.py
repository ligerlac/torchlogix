"""Whole-model properties, swept over the shared models in tests/models.py."""
import tempfile

import pytest
import torch

from torchlogix.layers import LearnableBinarization, LogicConv2d, LogicDense

from models import (
    MODELS,
    BranchModel,
    ConvModel,
    DenseModel,
    model_input,
)

# Variants with a LearnableBinarization front-end. Train/eval equivalence is
# only meaningful where there is something stochastic to collapse, so the
# binarized variants carry this test rather than the plain ones.
BINARIZED_MODELS = [
    pytest.param(lambda: DenseModel(binarize=True), id="dense-binarized"),
    pytest.param(lambda: ConvModel(binarize=True), id="conv-binarized"),
    pytest.param(BranchModel, id="branch"),
]


def _force_discrete(model):
    """Push every stochastic component to a near-deterministic extreme.

    With saturated gate weights and near-zero binarization temperatures the
    relaxed (train) forward should collapse onto the discrete (eval) one.
    """
    for module in model.modules():
        if isinstance(module, LearnableBinarization):
            module.temperature_sampling = 1e-9
            module.temperature_softplus = 1e-9

        if isinstance(module, LogicDense):
            # Give one randomly chosen gate per row an overwhelming weight.
            n_gates = module.weight.shape[0]
            indices = torch.randint(0, 16, (n_gates,), device=module.weight.device)
            rows = torch.arange(n_gates, device=module.weight.device)
            with torch.no_grad():
                module.weight[rows, indices] = 100

        if isinstance(module, LogicConv2d):
            for layer_weights in module.tree_weights:
                n_kernels, n_inputs = layer_weights.data.shape[:2]
                indices = torch.randint(
                    0, 16, (n_kernels, n_inputs, 1), device=layer_weights.device
                )
                with torch.no_grad():
                    layer_weights.scatter_(2, indices, 100)


@pytest.mark.parametrize("make_model", BINARIZED_MODELS)
def test_train_and_eval_agree_when_forced_discrete(make_model):
    """Relaxed and discrete forwards must agree once nothing is stochastic."""
    model = make_model()
    x = model_input(model, seed=0)

    _force_discrete(model)

    out_train = model(x)
    model.eval()
    out_eval = model(x)

    assert torch.allclose(out_eval, out_train, atol=1e-5), \
        "Eval-mode output diverges from train-mode output"


@pytest.mark.parametrize("model_cls", MODELS)
def test_state_dict_round_trip(model_cls):
    """A model reloaded from its own state_dict must produce identical output."""
    model = model_cls()
    model.eval()
    x = model_input(model, seed=0)
    out_original = model(x)

    reloaded = model_cls()
    reloaded.eval()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        reloaded.load_state_dict(torch.load(tmp.name))

    out_reloaded = reloaded(x)
    assert torch.allclose(out_original.float(), out_reloaded.float(), atol=1e-5), \
        "Output changed after a state_dict round trip"
