import torch
from torch.nn.functional import gumbel_softmax, softmax
from torch.distributions import Gumbel
import numpy as np
import random


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return softmax(y / temperature, dim=-1)

def gumbel_sigmoid_2(logits, tau=1.0, hard=False, threshold=0.5):
    """
    Fast Gumbel-Sigmoid implementation using logistic noise trick.
    """
    if tau <= 0:
        raise ValueError("Temperature must be positive")

    # Logistic(0,1) noise from uniform: log(U) - log(1-U)
    U = torch.rand_like(logits)
    logistic_noise = torch.log(U + 1e-20) - torch.log(1 - U + 1e-20)

    # Soft sample
    y_soft = torch.sigmoid((logits + logistic_noise) / tau)

    if hard:
        # Straight-through estimator
        y_hard = (y_soft > threshold).float()
        return (y_hard - y_soft).detach() + y_soft

    return y_soft


def gumbel_sigmoid_3(logits, tau=1.0, hard=False, threshold=0.5):
    """
    Fast Gumbel-Sigmoid implementation using logistic noise trick.
    """
    if tau <= 0:
        raise ValueError("Temperature must be positive")

    # Logistic(0,1) noise from uniform: log(U) - log(1-U)
    U = torch.rand_like(logits)
    gumbels = (
        -torch.empty((*logits.shape, 2), memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    logistic_noise = gumbels[..., 0] - gumbels[..., 1]

    # Soft sample
    y_soft = torch.sigmoid((logits + logistic_noise) / tau)

    if hard:
        # Straight-through estimator
        y_hard = (y_soft > threshold).float()
        return (y_hard - y_soft).detach() + y_soft

    return y_soft


def gumbel_sigmoid(logits, tau=1.0, hard=False, threshold=0.5):
    # Temperature must be positive.
    if tau <= 0:
        raise ValueError("Temperature must be positive")

    # Sample Gumbel noise. The difference of two Gumbels is equivalent to a Logistic distribution.
    gumbel_noise = Gumbel(0, 1).sample(logits.shape).to(logits.device) - \
                   Gumbel(0, 1).sample(logits.shape).to(logits.device)
    
    # Apply the reparameterization trick
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        # Straight-Through Estimator
        y_hard = (y_soft > threshold).float()
        return (y_hard - y_soft).detach() + y_soft
    
    return y_soft


class TestModule(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(n))

    def forward(self, x):
        y_ = x @ self.weights
        y = torch.softmax(y_.clone(), dim=-1)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        y_soft = gumbel_softmax(y_.clone(), tau=1, hard=False, dim=-1, eps=1e-10)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        y_hard = gumbel_softmax(y_.clone(), tau=1, hard=True, dim=-1)
        print(y.requires_grad)
        print(y_soft.detach().requires_grad)
        print(y_hard.requires_grad)
        return y

if __name__ == "__main__":
    n = 3
    x = torch.randn(2, 2, n)
    model = TestModule(n)
    y = model(x)
    loss = y.sum()
    loss.backward()
