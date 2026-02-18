import torch
import torch.nn as nn
from ..layers import OrPooling2d, GroupSum, LogicConv2d, LogicDense


class ResidualLogicBlockLiv(nn.Module):
    """
    I see what you did here. Creating a 1x1 conv for the skip connection
    by setting tree_depth=1, receptive_field_size=1, padding=0.
    But this still picks 2 random inputs along the channel axis for each kernel.
    I think we would also have to also set lut_rank=1 for the skip connection conv
    to make it a true 1x1 conv.
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
        lut_rank=2,
        **llkw,
    ):
        super().__init__()

        stride = 2 if downsample else 1

        self.main = nn.Sequential(
            LogicConv2d(
                in_dim=in_dim,
                channels=in_channels,
                num_kernels=out_channels,
                tree_depth=tree_depth,
                receptive_field_size=receptive_field_size,
                padding=padding,
                parametrization=parametrization,
                lut_rank=lut_rank,
                **llkw,
            ),
            OrPooling2d(kernel_size=2, stride=stride, padding=0) if downsample else nn.Identity(),
        )

        self.shortcut = nn.Identity()
        # we can either project the input to the output channels, or use a standard skip connection
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                LogicConv2d(
                    in_dim=in_dim,
                    channels=in_channels,
                    num_kernels=out_channels,
                    tree_depth=1,
                    receptive_field_size=1,
                    padding=0,
                    parametrization=parametrization,
                    **llkw,
                ),
                OrPooling2d(kernel_size=2, stride=stride, padding=0) if downsample else nn.Identity(),
            )

    def forward(self, x):
        out = self.main(x)
        identity = self.shortcut(x)
        # return out + identity
        return out + identity - out * identity  # Relaxed OR


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
