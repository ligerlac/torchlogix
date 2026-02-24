import torch
import torch.nn as nn
from ..layers import OrPooling2d, LogicConv2d


class ResidualLogicBlock(nn.Module):
    """
    This implements a resnet style block with two conv layers and a skip connection.
    The addition is replaced with a relaxed OR operation.
    """
    def __init__(
        self,
        in_dim,
        in_channels,
        out_channels,
        tree_depth=3,
        receptive_field_size=3,
        padding=1,
        downsample=False,
        parametrization="raw",
        **llkw,
    ):
        super().__init__()

        self.main = nn.Sequential(
            LogicConv2d(
                in_dim=in_dim,
                channels=in_channels,
                num_kernels=out_channels,
                tree_depth=tree_depth,
                receptive_field_size=receptive_field_size,
                padding=padding,
                parametrization=parametrization,
                **llkw,
            ),
            OrPooling2d(kernel_size=2, stride=2) if downsample else nn.Identity(),
            LogicConv2d(
                in_dim=in_dim//2 if downsample else in_dim,
                channels=out_channels,
                num_kernels=out_channels,
                tree_depth=tree_depth,
                receptive_field_size=receptive_field_size,
                padding=padding,
                parametrization=parametrization,
                **llkw,
            ),
            OrPooling2d(kernel_size=2, stride=2) if downsample else nn.Identity(),
        )

        shortcut_modules = []

        if downsample:
            shortcut_modules.append(OrPooling2d(kernel_size=4, stride=4))

        # Only add projection if needed
        if downsample or in_channels != out_channels:
            shortcut_modules.append(
                LogicConv2d(
                    in_dim=in_dim//4 if downsample else in_dim,
                    channels=in_channels,
                    num_kernels=out_channels,
                    tree_depth=0,
                    receptive_field_size=1,
                    padding=0,
                    parametrization=parametrization,
                    lut_rank=1,
                    **{k: llkw[k] for k in llkw if k != 'lut_rank'}
                )
            )

        if len(shortcut_modules) == 0:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(*shortcut_modules)


    def forward(self, x):
        out = self.main(x)
        identity = self.shortcut(x)
        return out + identity - out * identity  # Relaxed OR
