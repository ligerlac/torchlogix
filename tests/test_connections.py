import pytest
import torch
from torchlogix.layers import LogicDense
from torchlogix.connections import LearnableDenseConnections
from torchlogix.functional import softmax
from torch.nn.functional import softmax as softmax_torch


@pytest.mark.parametrize("parametrization", ["raw", "walsh", "light"])
@pytest.mark.parametrize("num_candidates", [-1, 1, 2, 3])
@pytest.mark.parametrize("lut_rank", [2, 4, 6])
def test_learnable_connections(parametrization, num_candidates, lut_rank):
    """Test that connections can be trained."""
    parametrization_kwargs = {
        "weight_init": "residual",
        "residual_param": 20.0
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
