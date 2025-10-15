import torch
import torch.nn as nn
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

class ResidualLogicBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_channels,
        tree_depth=3,
        receptive_field_size=3,
        padding=1,
        downsample=False,
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
                **llkw,
            ),
            OrPooling(kernel_size=2, stride=stride, padding=0) if downsample else nn.Identity(),
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
                    **llkw,
                ),
                OrPooling(kernel_size=2, stride=stride, padding=0) if downsample else nn.Identity(),
            )

    def forward(self, x):
        out = self.main(x)
        identity = self.shortcut(x)
        return out + identity

class ClgnMnist(torch.nn.Sequential):
    """
    Model as described in the paper 'Convolutional Logic Gate Networks'
    for the MNIST dataset.
    """

    def __init__(self, k_num: int=16, **llkw):
        super(ClgnMnist, self).__init__()
        self.k_num = k_num
        layers = []
        layers.append(
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
        layers.append(OrPooling(kernel_size=2, stride=2, padding=0))

        layers.append(
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
        layers.append(OrPooling(kernel_size=2, stride=2, padding=1))

        layers.append(
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
        layers.append(OrPooling(kernel_size=2, stride=2, padding=1))

        layers.append(torch.nn.Flatten())

        layers.append(LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, **llkw))
        layers.append(
            LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, **llkw)
        )
        layers.append(LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, **llkw))

        super(ClgnMnist, self).__init__(*layers, GroupSum(k=10, tau=1.0))


class ClgnMnistSmall(ClgnMnist):
    def __init__(self, **llkw):
        super(ClgnMnistSmall, self).__init__(k_num=16, **llkw)


class ClgnMnistMedium(ClgnMnist):
    def __init__(self, **llkw):
        super(ClgnMnistMedium, self).__init__(k_num=64, **llkw)


class ClgnMnistLarge(ClgnMnist):
    def __init__(self, **llkw):
        super(ClgnMnistLarge, self).__init__(k_num=1024, **llkw)


class ClgnCifar10(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 3-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """

    def __init__(self, n_bits: int, k_num: int, tau: float, **llkw):
        layers = []
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=k_num,
                channels=3*n_bits,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **llkw,
            )
        )
        layers.append(OrPooling(kernel_size=2, stride=2)) # kx16x16

        layers.append(
            LogicConv2d(
                in_dim=16,
                channels=k_num,
                num_kernels=4*k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **llkw,
            )
        )
        layers.append(OrPooling(kernel_size=2, stride=2)) # 4kx8x8

        layers.append(
            LogicConv2d(
                in_dim=8,
                channels=4*k_num,
                num_kernels=16*k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **llkw,
            )
        )
        layers.append(OrPooling(kernel_size=2, stride=2)) # 16kx4x4
        
        layers.append(
            LogicConv2d(
                in_dim=4,
                channels=16*k_num,
                num_kernels=32*k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **llkw,
            )
        )
        layers.append(OrPooling(kernel_size=2, stride=2)) # 32kx2x2

        layers.append(torch.nn.Flatten()) # 128k

        layers.append(LogicDense(in_dim=128*k_num, out_dim=1280*k_num, **llkw))
        layers.append(LogicDense(in_dim=1280*k_num, out_dim=640*k_num, **llkw))
        layers.append(LogicDense(in_dim=640*k_num, out_dim=320*k_num, **llkw))

        super(ClgnCifar10, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10Res(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 3-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """

    def __init__(self, n_bits: int, k_num: int, tau: float, **llkw):
        layers = []

        layers.append(
            ResidualLogicBlock(
                in_dim=32,
                in_channels=3*n_bits,
                out_channels=2*k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                downsample=True,
                **llkw,
            )
        )
    
        layers.append(torch.nn.Flatten()) # 4x4x16k = 256k

        layers.append(LogicDense(in_dim=256*2*k_num, out_dim=512*k_num, **llkw))
        layers.append(LogicDense(in_dim=512*k_num, out_dim=256*k_num, **llkw))
        layers.append(LogicDense(in_dim=256*k_num, out_dim=320*k_num, **llkw))

        super(ClgnCifar10Res, self).__init__(*layers, GroupSum(k=10, tau=tau))

class ClgnCifar10SmallRes(ClgnCifar10Res):
    def __init__(self, **llkw):
        super(ClgnCifar10SmallRes, self).__init__(n_bits=3, k_num=32, tau=20, **llkw)


class ClgnCifar10Small(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Small, self).__init__(n_bits=3, k_num=32, tau=20, **llkw)


class ClgnCifar10Medium(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Medium, self).__init__(n_bits=3, k_num=256, tau=40, **llkw)

class ClgnCifar10MediumRes(ClgnCifar10Res):
    def __init__(self, **llkw):
        super(ClgnCifar10MediumRes, self).__init__(n_bits=3, k_num=256, tau=40, **llkw)



class ClgnCifar10Large(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Large, self).__init__(n_bits=5, k_num=512, tau=280, **llkw)


class ClgnCifar10Large2(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Large2, self).__init__(n_bits=5, k_num=1024, tau=340, **llkw)


class ClgnCifar10Large4(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Large4, self).__init__(n_bits=5, k_num=2560, tau=450, **llkw)


class ClgnCifar10Tiny(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    Takes 3-bit-thresholded inputs. 
    """

    def __init__(self, k_num=64, **llkw):
        n_bits = 3
        tau = 20
        layers = []
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=k_num,
                channels=3*n_bits,
                tree_depth=3,
                receptive_field_size=5,
                **llkw,
            )
        ) # kx28x28
        layers.append(OrPooling(kernel_size=2, stride=2)) # kx14x14

        layers.append(
            LogicConv2d(
                in_dim=14,
                channels=k_num,
                num_kernels=4*k_num,
                tree_depth=3,
                receptive_field_size=3,
                **llkw,
            )
        )  # 4kx12x12
        # layers.append(OrPooling(kernel_size=2, stride=2)) # 4kx6x6

        layers.append(torch.nn.Flatten()) # 4kx6x6=144k

        # layers.append(LogicDense(in_dim=144*k_num, out_dim=1280*k_num, **llkw))
        layers.append(LogicDense(in_dim=576*k_num, out_dim=1280*k_num, **llkw))
        layers.append(LogicDense(in_dim=1280*k_num, out_dim=640*k_num, **llkw))
        layers.append(LogicDense(in_dim=640*k_num, out_dim=320*k_num, **llkw))

        super(ClgnCifar10Tiny, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10Tiny32(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        super(ClgnCifar10Tiny32, self).__init__(k_num=32, **llkw)

class ClgnCifar10Tiny64(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        super(ClgnCifar10Tiny64, self).__init__(k_num=64, **llkw)

class ClgnCifar10Tiny128(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        super(ClgnCifar10Tiny128, self).__init__(k_num=128, **llkw)

class ClgnCifar10Tiny256(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        super(ClgnCifar10Tiny256, self).__init__(k_num=256, **llkw)


class ClgnCifar10Mini(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    Takes continuous (unthresholded) inputs. Only single conv layer.
    """

    def __init__(self, k_num=64, tau=10, **llkw):
        n_bits = 3
        tau = 20
        layers = []
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=k_num,
                channels=3*n_bits,
                tree_depth=3,
                receptive_field_size=4,
                **llkw,
            )
        ) # kx28x28

        layers.append(torch.nn.Flatten()) # kx29x29=841k

        layers.append(LogicDense(in_dim=841*k_num, out_dim=512*k_num, **llkw))
        layers.append(LogicDense(in_dim=512*k_num, out_dim=256*k_num, **llkw))
        layers.append(LogicDense(in_dim=256*k_num, out_dim=128*k_num, **llkw))
        layers.append(LogicDense(in_dim=128*k_num, out_dim=60*k_num, **llkw))

        super(ClgnCifar10Mini, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10Mini32(ClgnCifar10Mini):
    def __init__(self, **llkw):
        super(ClgnCifar10Mini32, self).__init__(k_num=32, tau=5., **llkw)

class ClgnCifar10Mini64(ClgnCifar10Mini):
    def __init__(self, **llkw):
        super(ClgnCifar10Mini64, self).__init__(k_num=64, tau=10., **llkw)

class ClgnCifar10Mini128(ClgnCifar10Mini):
    def __init__(self, **llkw):
        super(ClgnCifar10Mini128, self).__init__(k_num=128, tau=20., **llkw)

class ClgnCifar10Mini256(ClgnCifar10Mini):
    def __init__(self, **llkw):
        super(ClgnCifar10Mini256, self).__init__(k_num=256, tau=40., **llkw)
