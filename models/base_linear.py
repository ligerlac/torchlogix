import torch
from models.difflog_layers.linear import LogicLayer, GroupSum

class BaseModel(torch.nn.Module):
    def __init__(self, in_dim, k, l, class_count, args, **llkw):
        super(BaseModel, self).__init__()
        logic_layers = []
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

        self.model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    def forward(self, x):
        return self.model(x)
