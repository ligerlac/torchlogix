import torch
from torch.nn.functional import gumbel_softmax, softmax
import numpy as np
import random


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return softmax(y / temperature, dim=-1)


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
