import torch
import torch.nn as nn
from ..layers import OrPooling2d, GroupSum, LogicConv2d, LogicDense
from ..layers.binarization import setup_binarization
from ..modules.resblock import ResidualLogicBlock, ResidualLogicBlockLiv


class ClgnMnist(torch.nn.Sequential):
    """
    Model as described in the paper 'Convolutional Logic Gate Networks'
    for the MNIST dataset.
    """

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, 
                 k_num: int=16, parametrization="raw", tau=1.0, **llkw):
        
        binarization = "dummy"
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)
        self.k_num = k_num
        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=28,
                num_kernels=k_num,
                channels=1,
                **llkw,
                tree_depth=3,
                receptive_field_size=5,
                padding=0,
                parametrization=parametrization,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2, padding=0))

        layers.append(
            LogicConv2d(
                in_dim=12,
                channels=k_num,
                num_kernels=3 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
                parametrization=parametrization,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        layers.append(
            LogicConv2d(
                in_dim=6,
                channels=3 * k_num,
                num_kernels=9 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
                parametrization=parametrization,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        layers.append(torch.nn.Flatten())

        layers.append(LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, parametrization=parametrization, **llkw))

        super(ClgnMnist, self).__init__(*layers, GroupSum(k=10, tau=tau))

class ClgnMnistTiny(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1.0)
        super(ClgnMnistTiny, self).__init__(k_num=4, tau=tau, **llkw)


class ClgnMnistSmall(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 6.5)
        super(ClgnMnistSmall, self).__init__(k_num=16, tau=tau, **llkw)


class ClgnMnistMedium(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 28.)
        super(ClgnMnistMedium, self).__init__(k_num=64, tau=tau, **llkw)


class ClgnMnistLarge(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 35.)
        super(ClgnMnistLarge, self).__init__(k_num=1024, tau=tau, **llkw)


class ClgnCifar10(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 2-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """
    n_input_bits = None
    k_num = None
    tau = None

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, connections_kwargs: dict, **llkw):
        assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."
        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        # if "coonections_kwargs" in llkw:
        #     llkw["connections_kwargs"] = 

        connections_kwargs = dict(connections_kwargs)  # make a copy to avoid modifying the original
        connections_kwargs["channel_group_size"] = 2  # from the paper, we use grouped connections with channel group size 2 for conv layers

        print(f"connctions_kwargs for {self.__class__.__name__}: {connections_kwargs}")

        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=self.k_num,
                channels=3*n_bits,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # kx16x16

        layers.append(
            LogicConv2d(
                in_dim=16,
                channels=self.k_num,
                num_kernels=4*self.k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 4kx8x8

        layers.append(
            LogicConv2d(
                in_dim=8,
                channels=4*self.k_num,
                num_kernels=16*self.k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 16kx4x4
        
        layers.append(
            LogicConv2d(
                in_dim=4,
                channels=16*self.k_num,
                num_kernels=32*self.k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 32kx2x2

        layers.append(torch.nn.Flatten()) # 128k

        layers.append(LogicDense(in_dim=128*self.k_num, out_dim=1280*self.k_num, **llkw))
        layers.append(LogicDense(in_dim=1280*self.k_num, out_dim=640*self.k_num, **llkw))
        layers.append(LogicDense(in_dim=640*self.k_num, out_dim=320*self.k_num, **llkw))

        super(ClgnCifar10, self).__init__(*layers, GroupSum(k=10, tau=self.tau))


class ClgnCifar10Small(ClgnCifar10):
    n_input_bits = 2
    k_num = 32
    tau = 20


class ClgnCifar10Medium(ClgnCifar10):
    n_input_bits = 2
    k_num = 256
    tau = 40


class ClgnCifar10Large(ClgnCifar10):
    n_input_bits = 5
    k_num = 512
    tau = 280


class ClgnCifar10Large2(ClgnCifar10):
    n_input_bits = 5
    k_num = 1024
    tau = 340


class ClgnCifar10Large4(ClgnCifar10):
    n_input_bits = 5
    k_num = 2560
    tau = 450
