import pytest
import torch
from torchlogix.layers import LogicDense
from torchlogix.connections import LearnableDenseConnections, FixedConvConnections
from torchlogix.functional import softmax
from torchlogix.utils import set_export_mode
from torch.fx.experimental.proxy_tensor import make_fx
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


# ---------------------------------------------------------------------------
# Train / eval / export modes
#
# Learnable connections pick their wiring with an argmax over the candidate
# axis, optionally perturbed by Gumbel noise. The noise is a training-time
# exploration device, and neither it nor the argmax may survive into the
# exported graph - the circuit builder expects wiring to be a plain gather
# with a constant index tensor.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gumbel", [False, True])
def test_learnable_connections_eval_is_deterministic(gumbel):
    """eval() must not sample Gumbel noise."""
    torch.manual_seed(0)
    layer = LogicDense(16, 8, connections="learnable",
                       connections_kwargs={"num_candidates": 2, "gumbel": gumbel})
    layer.eval()
    x = (torch.rand(4, 16) > 0.5).float()

    first = layer(x)
    for _ in range(4):
        assert torch.equal(layer(x), first)


def test_learnable_connections_train_uses_gumbel():
    """The counterpart: in train() the noise is live, so wiring may vary.

    Guards against "fixing" the eval determinism by dropping Gumbel entirely.
    """
    torch.manual_seed(0)
    conns = LearnableDenseConnections(in_dim=64, out_dim=64, num_candidates=8, gumbel=True)
    conns.train()
    x = (torch.rand(4, 64) > 0.5).float()

    outs = [conns(x) for _ in range(20)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), (
        "expected Gumbel sampling to vary the wiring during training"
    )


@pytest.mark.parametrize("gumbel", [False, True])
def test_learnable_connections_export_matches_eval(gumbel):
    torch.manual_seed(0)
    layer = LogicDense(16, 8, connections="learnable",
                       connections_kwargs={"num_candidates": 2, "gumbel": gumbel})
    layer.eval()
    x = (torch.rand(4, 16) > 0.5)

    expected = layer(x.float())
    layer.set_export_mode(True)
    assert torch.equal(layer(x).bool(), expected.bool())


@pytest.mark.parametrize("via", ["utils_walk", "method"])
def test_set_export_mode_reaches_connections(via):
    """Connections must enter export mode too.

    Both routes have to work: the module-level walk, and calling the method on
    a layer directly - the latter is recursive, like nn.Module.eval().
    """
    torch.manual_seed(0)
    layer = LogicDense(16, 8, connections="learnable",
                       connections_kwargs={"num_candidates": 2})
    enable = (lambda on: set_export_mode(layer, on)) if via == "utils_walk" \
        else (lambda on: layer.set_export_mode(on))

    enable(True)
    assert layer.connections.export_mode is True
    assert layer.connections._export_indices.shape == (layer.lut_rank, layer.out_dim)

    enable(False)
    assert layer.connections.export_mode is False
    assert not hasattr(layer.connections, "_export_indices")


@pytest.mark.parametrize("connections, kwargs", [
    ("fixed", {}),
    ("learnable", {"num_candidates": 2}),
])
def test_export_mode_is_compilable(connections, kwargs):
    """The export forward must survive torch.compile for either wiring."""
    torch.manual_seed(0)
    layer = LogicDense(16, 8, connections=connections, connections_kwargs=kwargs)
    layer.set_export_mode(True)
    x = (torch.rand(4, 16) > 0.5)

    assert torch.equal(torch.compile(layer)(x), layer(x))


def test_resolved_connections_match_argmax():
    """The frozen export wiring is the argmax over the candidate axis."""
    torch.manual_seed(0)
    conns = LearnableDenseConnections(in_dim=16, out_dim=8, num_candidates=3)

    resolved = conns.resolve_connections()
    chosen = conns.weights.argmax(dim=0)
    for l in range(conns.lut_rank):
        for o in range(conns.out_dim):
            assert resolved[l, o] == conns.indices[chosen[l, o], l, o]


@pytest.mark.parametrize("gumbel", [False, True])
def test_learnable_export_graph_is_as_clean_as_fixed(gumbel):
    """In export mode learnable wiring must trace to the same ops as fixed wiring."""
    def traced_ops(connections, connections_kwargs):
        torch.manual_seed(0)
        layer = LogicDense(16, 8, connections=connections,
                           connections_kwargs=connections_kwargs)
        set_export_mode(layer, True)
        x = (torch.rand(1, 16) > 0.5)
        gm = make_fx(layer)(x)
        return {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}

    learnable = traced_ops("learnable", {"num_candidates": 2, "gumbel": gumbel})
    fixed = traced_ops("fixed", {})

    assert learnable == fixed, f"extra ops in learnable graph: {sorted(learnable - fixed)}"
