import torch

from ..layers import GroupSum, LogicDense


class RandomlyConnectedNN(torch.nn.Module):
    """
    A difflog nn where node connections are random.

    All nodes get two binary inputs and one output.
    """

    def __init__(self, in_dim, k, layers, class_count, tau, **llkw):
        super(RandomlyConnectedNN, self).__init__()
        logic_layers = []
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicDense(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(layers - 1):
            logic_layers.append(LogicDense(in_dim=k, out_dim=k, **llkw))

        self.model = torch.nn.Sequential(*logic_layers, GroupSum(class_count, tau))

    def forward(self, x):
        """Forward pass of the randomly connected neural network."""
        return self.model(x)
