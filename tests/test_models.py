"""Whole-model properties, swept over the shared models in tests/models.py.

Covers train/eval agreement, state_dict round-trips, gradient correctness,
eval-vs-export agreement, and that an exported graph contains nothing but
logic and view ops. Adding a model to MODELS gives it all of them.
"""
import operator
import tempfile

import pytest
import torch

from torchlogix.layers import GroupSum, LearnableBinarization, LogicConv2d, LogicDense
from torchlogix.utils import set_export_mode

from helpers import (
    assert_compiled_gradients_match_eager,
    assert_finite_difference_matches_autograd,
    model_input,
    random_bool_input,
)
from models import (
    BINARIZED_MODELS,
    EXACT_GRADIENT_MODELS,
    MODELS,
    TORCHLOGIX_MODELS,
    BranchModel,
    ConvModel,
)


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


@pytest.mark.parametrize("model_cls", BINARIZED_MODELS)
def test_train_and_eval_agree_when_forced_discrete(model_cls):
    """Relaxed and discrete forwards must agree once nothing is stochastic."""
    model = model_cls()
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

# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------

def _train_mode_batch(model_cls, seed=0):
    torch.manual_seed(seed)
    model = model_cls()
    model.train()
    x = model_input(model, batch_size=2, seed=seed)
    torch.manual_seed(seed + 1)
    weights = torch.rand_like(model(x).float())
    return model, x, weights


@pytest.mark.parametrize("model_cls", EXACT_GRADIENT_MODELS)
def test_model_gradients_match_finite_differences(model_cls):
    """Autograd must agree with central finite differences through a whole model.

    Sweeps EXACT_GRADIENT_MODELS rather than TORCHLOGIX_MODELS: models with
    learnable connections train through a discrete argmax with a surrogate
    estimator, where finite differences are expected to disagree. See the note
    on EXACT_GRADIENT_MODELS in models.py.
    """
    model, x, weights = _train_mode_batch(model_cls)
    assert_finite_difference_matches_autograd(model, x, weights)


@pytest.mark.slow
@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
def test_model_gradients_survive_torch_compile(model_cls):
    """Compiling a model must not change its gradients.

    Marked slow: torch.compile costs a few seconds per model. Two
    representative models rather than all of them, since the per-layer version
    in test_layers.py already covers every layer family.
    """
    model, x, weights = _train_mode_batch(model_cls)
    assert_compiled_gradients_match_eager(model, x, weights)


# ---------------------------------------------------------------------------
# Eval mode vs export mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", MODELS)
def test_eval_and_export_agree(model_cls):
    """Eval mode and export mode must agree on binary inputs."""
    model = model_cls()
    model.eval()
    x = random_bool_input(model, batch_size=8, seed=0)

    result_eval = model(x if model.input_dtype == torch.bool else x.float())

    set_export_mode(model)
    result_export = model(x)

    assert torch.allclose(result_eval.float(), result_export.float(), atol=1e-6), \
        f"[{model_cls.__name__}] eval and export results diverge"


# ---------------------------------------------------------------------------
# Exported graph purity
#
# An exported model must lower to logic and view ops only. The inventory below
# is the allow-list; anything else in the graph means a real tensor op survived
# export and the model would not be convertible to a circuit.
# ---------------------------------------------------------------------------

ALLOWED_FX_TARGETS = {
    # Logic ops — dunder and explicit bitwise forms (both may appear after export lowering)
    torch.ops.aten.__and__.Tensor,
    torch.ops.aten.__or__.Tensor,
    torch.ops.aten.__xor__.Tensor,
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.bitwise_or.Tensor,
    torch.ops.aten.bitwise_xor.Tensor,
    torch.ops.aten.bitwise_not.default,

    # Alias — emitted for identity wire ops (WIRE A / WIRE B) in native decomposition
    torch.ops.aten.alias.default,

    # LUT ops (kept for backward compat; not emitted with native ops)
    torch.ops.aten.where.self,
    torch.ops.aten.eq.Scalar,

    # Comparisons (needed for export guards)
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.le.Scalar,
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.lt.Scalar,
    operator.ge,
    operator.le,
    operator.gt,
    operator.lt,
    operator.getitem,

    # Indexing / wiring
    torch.ops.aten.index.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.unbind.int,

    # Shape / layout (view ops)
    torch.ops.aten.reshape.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.moveaxis.int,
    torch.ops.aten.movedim.int,   # alias of moveaxis; emitted by transposed-conv dilation
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.pad.default,
    torch.ops.aten.unfold.default,
    torch.ops.aten._unsafe_view.default,

    # Advanced view variants
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.stack.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.chunk.default,

    # Index writes
    torch.ops.aten.index_put_.default,
    torch.ops.aten.index_put.default,

    # Constants and copies
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.ones_like.default,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.lift_fresh_copy.default,
    torch.ops.aten.clone.default,

    # Symbolic shape system (export internals)
    torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_constrain_range_for_size.default,
    torch.ops.aten._assert_scalar.default,
}

ALLOWED_FX_TARGETS_GROUP_SUM = {
    # Native decomposition of group_sum: reshape + sum + float + optional scale
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.to.dtype,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten._assert_tensor_metadata.default,
}


@pytest.mark.parametrize("model_cls", TORCHLOGIX_MODELS)
def test_exported_graph_is_pure_logic(model_cls):
    model = model_cls()
    model.eval()

    # Allow the reduction ops only for models that actually reduce, so the
    # check stays tight for the ones whose output is raw logic.
    allowed_targets = ALLOWED_FX_TARGETS
    if any(isinstance(m, GroupSum) for m in model.modules()):
        allowed_targets = allowed_targets | ALLOWED_FX_TARGETS_GROUP_SUM

    x = random_bool_input(model, batch_size=8, seed=0)
    set_export_mode(model)

    gm = torch.export.export(model, (x,), strict=False).module()

    disallowed = [
        f"{node.name}: {node.target}"
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target not in allowed_targets
    ]
    assert not disallowed, (
        "exported graph contains non-logic ops:\n" + "\n".join(disallowed)
    )


def test_exported_graph_without_group_sum_contains_no_reductions():
    """With the GroupSum head removed, nothing but logic and view ops may remain.

    The parametrized test above widens the allow-list for models that reduce.
    This is the strict case: no reduction ops are permitted at all.
    """
    model = ConvModel()
    del model[-1]           # drop the trailing GroupSum
    model.eval()

    x = random_bool_input(model, batch_size=8, seed=0)
    set_export_mode(model)
    gm = torch.export.export(model, (x,), strict=False).module()

    disallowed = [
        f"{node.name}: {node.target}"
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target not in ALLOWED_FX_TARGETS
    ]
    assert not disallowed, (
        "graph without GroupSum still contains non-logic ops:\n" + "\n".join(disallowed)
    )
