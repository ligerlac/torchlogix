import torch
from models.difflog_layers.linear import LogicLayer, GroupSum

class RandomlyConnectedNN(torch.nn.Module):
    def __init__(self, in_dim, k, l, class_count, tau, **llkw):
        super(RandomlyConnectedNN, self).__init__()
        logic_layers = []
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

        self.model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, tau)
        )

    def forward(self, x):
        return self.model(x)
