import torch

from ..layers import LogicDense, LogicConv2d, OrPooling, GroupSum


class CNN(torch.nn.Module):
    """An implementation of a logic gate convolutional neural network."""

    def __init__(self, class_count, tau, **llkw):
        super(CNN, self).__init__()
        logic_layers = []
        # specifically written for mnist
        k_num = 16
        logic_layers.append(
            LogicConv2d(
                in_dim=28,
                num_kernels=k_num,
                channels=1,
                **llkw,
                tree_depth=3,
                receptive_field_size=5,
                padding=0,
            )
        )
        logic_layers.append(OrPooling(kernel_size=2, stride=2, padding=0))

        logic_layers.append(
            LogicConv2d(
                in_dim=12,
                channels=k_num,
                num_kernels=3 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
            )
        )
        logic_layers.append(OrPooling(kernel_size=2, stride=2, padding=1))

        logic_layers.append(
            LogicConv2d(
                in_dim=6,
                channels=3 * k_num,
                num_kernels=9 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
            )
        )
        logic_layers.append(OrPooling(kernel_size=2, stride=2, padding=1))

        logic_layers.append(torch.nn.Flatten())

        logic_layers.append(LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, **llkw))
        logic_layers.append(
            LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, **llkw)
        )
        logic_layers.append(LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, **llkw))

        self.model = torch.nn.Sequential(*logic_layers, GroupSum(class_count, tau))

    def forward(self, x):
        """Forward pass of the logic gate convolutional neural network."""
        return self.model(x)
