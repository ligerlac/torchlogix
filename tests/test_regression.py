from torchlogix.layers.groupsum import LearnableGroupAffine, LearnableGroupLinear
import torch


def test_affine():
    k = 2
    init_a = 0.0
    module = LearnableGroupAffine(k=k, init_a=init_a, init_b=0.0)
    input_tensor = torch.tensor([
        [0, 0, 0, 0], 
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=torch.float32)
    output = module(input_tensor)
    expected_output = torch.tensor([
        [-1.0, -1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], dtype=torch.float32)
    assert torch.allclose(output, expected_output)


def test_linear():
    k = 2
    module = LearnableGroupLinear(k=k)
    input_tensor = torch.tensor([
        [0, 0, 0, 0], 
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=torch.float32)
    output = module(input_tensor)
    # Since the linear layer is randomly initialized, we can't check for exact values,
    # but we can check the shape and that the output is finite.
    assert output.shape == (4, k)
    assert torch.isfinite(output).all()
