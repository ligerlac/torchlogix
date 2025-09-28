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


class ClgnCifar10(torch.nn.Module):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 3-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """

    def __init__(self, n_bits: int, k_num: int, tau: float, **llkw):
        super(ClgnCifar10, self).__init__()
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


class ClgnCifar10Small(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Small, self).__init__(n_bits=2, k_num=32, tau=20, **llkw)


class ClgnCifar10Medium(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Medium, self).__init__(n_bits=2, k_num=256, tau=40, **llkw)


class ClgnCifar10Large(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Large, self).__init__(n_bits=5, k_num=512, tau=280, **llkw)


class ClgnCifar10Large2(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Large2, self).__init__(n_bits=5, k_num=1024, tau=340, **llkw)


class ClgnCifar10Large4(ClgnCifar10):
    def __init__(self, **llkw):
        super(ClgnCifar10Large4, self).__init__(n_bits=5, k_num=2560, tau=450, **llkw)
