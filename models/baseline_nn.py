import torch

class FullyConnectedNN(torch.nn.Module):
    def __init__(self, in_dim, k, l, class_count, dtype):
        super(FullyConnectedNN, self).__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(in_dim, k, dtype=dtype))
        layers.append(torch.nn.ReLU())
        for _ in range(l - 2):
            layers.append(torch.nn.Linear(k, k, dtype=dtype))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(k, class_count, dtype=dtype))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x