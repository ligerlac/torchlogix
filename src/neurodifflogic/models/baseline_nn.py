"""Baseline neural network implementation."""

import torch


class FullyConnectedNN(torch.nn.Module):
    """Fully connected neural network - used for benchmarking."""

    def __init__(self, in_dim, k, layers, class_count, dtype):
        super(FullyConnectedNN, self).__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(in_dim, k, dtype=dtype))
        layers.append(torch.nn.ReLU())
        for _ in range(layers - 2):
            layers.append(torch.nn.Linear(k, k, dtype=dtype))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(k, class_count, dtype=dtype))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the fully connected neural network."""
        x = self.model(x)
        return x
