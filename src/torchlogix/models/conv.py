import torch
import torch.nn as nn
from ..layers import OrPooling2d, GroupSum, LogicConv2d, LogicDense
from ..layers.binarization import setup_binarization
from ..modules.resblock import ResidualLogicBlock, ResidualLogicBlockLiv


class CNN(torch.nn.Module):
    """An implementation of a logic gate convolutional neural network."""

    def __init__(self, class_count, tau, parametrization="raw", **llkw):
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
                parametrization=parametrization,
                padding=0,
            )
        )
        logic_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=0))

        logic_layers.append(
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
        logic_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        logic_layers.append(
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
        logic_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        logic_layers.append(torch.nn.Flatten())

        logic_layers.append(LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, parametrization=parametrization, **llkw))
        logic_layers.append(LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, parametrization=parametrization, **llkw))
        logic_layers.append(LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, parametrization=parametrization, **llkw))

        self.model = torch.nn.Sequential(*logic_layers, GroupSum(class_count, tau))

    def forward(self, x):
        """Forward pass of the logic gate convolutional neural network."""
        return self.model(x)


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


class ClgnMnistSmallRank4(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 6.5)
        super(ClgnMnistSmallRank4, self).__init__(k_num=8, tau=tau, **llkw)


class ClgnMnistSmallRank6(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 6.5)
        super(ClgnMnistSmallRank6, self).__init__(k_num=6, tau=tau, **llkw)


class ClgnCifar10(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 2-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """
    conv_lut_rank = 2
    dense_lut_rank = 2
    tree_depth = 3
    n_input_bits = None
    conv_parametrization = None
    dense_parametrization = None
    
    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict,
                 k_num: int, tau: float, parametrization="raw", **llkw):
        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."
        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        del llkw['lut_rank']  # remove lut_rank from llkw to avoid conflict

        assert (self.conv_parametrization is None) == (self.dense_parametrization is None), "conv and dense parametrization must be both None or both set."
        self._conv_parametrization = self.conv_parametrization if self.conv_parametrization is not None else parametrization
        self._dense_parametrization = self.dense_parametrization if self.dense_parametrization is not None else parametrization
        if 'parametrization' in llkw:
            del llkw['parametrization']  # remove parametrization from llkw to avoid conflict

        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=k_num,
                channels=3*n_bits,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                parametrization=self._conv_parametrization,
                lut_rank=self.conv_lut_rank,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # kx16x16

        layers.append(
            LogicConv2d(
                in_dim=16,
                channels=k_num,
                num_kernels=4*k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                parametrization=self._conv_parametrization,
                lut_rank=self.conv_lut_rank,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 4kx8x8

        layers.append(
            LogicConv2d(
                in_dim=8,
                channels=4*k_num,
                num_kernels=16*k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                parametrization=self._conv_parametrization,
                lut_rank=self.conv_lut_rank,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 16kx4x4
        
        layers.append(
            LogicConv2d(
                in_dim=4,
                channels=16*k_num,
                num_kernels=32*k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                parametrization=self._conv_parametrization,
                lut_rank=self.conv_lut_rank,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 32kx2x2

        layers.append(torch.nn.Flatten()) # 128k

        layers.append(LogicDense(in_dim=128*k_num, out_dim=1280*k_num, parametrization=self._dense_parametrization, lut_rank=self.dense_lut_rank, **llkw))
        layers.append(LogicDense(in_dim=1280*k_num, out_dim=640*k_num, parametrization=self._dense_parametrization, lut_rank=self.dense_lut_rank, **llkw))
        layers.append(LogicDense(in_dim=640*k_num, out_dim=320*k_num, parametrization=self._dense_parametrization, lut_rank=self.dense_lut_rank, **llkw))

        super(ClgnCifar10, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10Res(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 3-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """
    conv_lut_rank = 2
    dense_lut_rank = 2
    tree_depth = 3
    n_input_bits = None


    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict,
                 k_num: int, tau: float, parametrization="raw", **llkw):
        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."
        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        del llkw['lut_rank']  # remove lut_rank from llkw to avoid conflict

        layers = [binarization_module]
        layers.append(
            ResidualLogicBlock(
                in_dim=32,
                in_channels=3*n_bits,
                out_channels=4*k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                downsample=True,
                parametrization=parametrization,
                lut_rank=self.conv_lut_rank,
                **llkw,
            )
        )
        layers.append(
            ResidualLogicBlock(
                in_dim=8,
                in_channels=2*k_num,
                out_channels=32*k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                downsample=True,
                parametrization=parametrization,
                lut_rank=self.conv_lut_rank,
                **llkw,
            )
        )
        layers.append(torch.nn.Flatten()) # 4x4x16k = 256k

        layers.append(LogicDense(in_dim=128*k_num, out_dim=1280*k_num, parametrization=parametrization, lut_rank=self.dense_lut_rank, **llkw))
        layers.append(LogicDense(in_dim=1280*k_num, out_dim=640*k_num, parametrization=parametrization, lut_rank=self.dense_lut_rank, **llkw))
        layers.append(LogicDense(in_dim=640*k_num, out_dim=320*k_num, parametrization=parametrization, lut_rank=self.dense_lut_rank, **llkw))

        super(ClgnCifar10Res, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10SmallRes(ClgnCifar10Res):
    n_input_bits = 2
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10SmallRes, self).__init__(k_num=32, tau=tau, **llkw)


class ClgnCifar10Small(ClgnCifar10):
    conv_lut_rank = 2
    dense_lut_rank = 2
    n_input_bits = 2
    tree_depth = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10Small, self).__init__(k_num=32, tau=tau, **llkw)


class ClgnCifar10SmallRank4(ClgnCifar10):
    conv_lut_rank = 4
    dense_lut_rank = 4
    n_input_bits = 2
    tree_depth = 1
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10SmallRank4, self).__init__(k_num=32, tau=tau, **llkw)


class ClgnCifar10SmallRank6(ClgnCifar10):
    conv_lut_rank = 6
    dense_lut_rank = 6
    n_input_bits = 2
    tree_depth = 1
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10SmallRank6, self).__init__(k_num=32, tau=tau, **llkw)


class ClgnCifar10SmallRankMixed(ClgnCifar10):
    conv_lut_rank = 4
    dense_lut_rank = 2
    n_input_bits = 2
    tree_depth = 1
    conv_parametrization = "warp"
    dense_parametrization = "raw"
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10SmallRankMixed, self).__init__(k_num=32, tau=tau, **llkw)


class ClgnCifar10Medium(ClgnCifar10):
    n_input_bits = 2
    def __init__(self, **llkw):
        tau = llkw.get("tau", 40)
        super(ClgnCifar10Medium, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10Medium(ClgnCifar10):
    conv_lut_rank = 2
    dense_lut_rank = 2
    n_input_bits = 2
    tree_depth = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 40)
        super(ClgnCifar10Medium, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10MediumRank4(ClgnCifar10):
    conv_lut_rank = 4
    dense_lut_rank = 4
    n_input_bits = 2
    tree_depth = 1
    def __init__(self, **llkw):
        tau = llkw.get("tau", 40)
        super(ClgnCifar10MediumRank4, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10MediumRankMixed(ClgnCifar10):
    conv_lut_rank = 4
    dense_lut_rank = 2
    n_input_bits = 2
    tree_depth = 1
    conv_parametrization = "warp"
    dense_parametrization = "raw"
    def __init__(self, **llkw):
        tau = llkw.get("tau", 40)
        super(ClgnCifar10MediumRankMixed, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10MediumRes(ClgnCifar10Res):
    n_input_bits = 2
    def __init__(self, **llkw):
        tau = llkw.get("tau", 40)
        super(ClgnCifar10MediumRes, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10Large(ClgnCifar10):
    n_input_bits = 5
    def __init__(self, **llkw):
        tau = llkw.get("tau", 280)
        super(ClgnCifar10Large, self).__init__(k_num=512, tau=tau, **llkw)


class ClgnCifar10Large2(ClgnCifar10):
    n_input_bits = 5
    def __init__(self, **llkw):
        tau = llkw.get("tau", 340)
        super(ClgnCifar10Large2, self).__init__(k_num=1024, tau=tau, **llkw)


class ClgnCifar10Large4(ClgnCifar10):
    n_input_bits = 5
    def __init__(self, **llkw):
        tau = llkw.get("tau", 450)
        super(ClgnCifar10Large4, self).__init__(k_num=2560, tau=tau, **llkw)


class ClgnCifar10Tiny(torch.nn.Sequential):
    conv_lut_rank = 2
    dense_lut_rank = 2
    tree_depth = 3
    n_input_bits = None
    conv_parametrization = None
    dense_parametrization = None
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    Takes 3-bit-thresholded inputs. 
    """

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, 
                 k_num=64, parametrization="raw", tau=20, **llkw):
        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."        
        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        del llkw['lut_rank']  # remove lut_rank from llkw to avoid conflict

        assert (self.conv_parametrization is None) == (self.dense_parametrization is None), "conv and dense parametrization must be both None or both set."
        self._conv_parametrization = self.conv_parametrization if self.conv_parametrization is not None else parametrization
        self._dense_parametrization = self.dense_parametrization if self.dense_parametrization is not None else parametrization
        if 'parametrization' in llkw:
            del llkw['parametrization']  # remove parametrization from llkw to avoid conflict

        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=k_num,
                channels=3*n_bits,
                tree_depth=3,
                receptive_field_size=5,
                parametrization=parametrization,
                **llkw,
            )
        ) # kx28x28
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # kx14x14

        layers.append(
            LogicConv2d(
                in_dim=14,
                channels=k_num,
                num_kernels=4*k_num,
                tree_depth=3,
                receptive_field_size=3,
                parametrization=parametrization,
                **llkw,
            )
        )  # 4kx12x12
        # layers.append(OrPooling2d(kernel_size=2, stride=2)) # 4kx6x6

        layers.append(torch.nn.Flatten()) # 4kx6x6=144k

        # layers.append(LogicDense(in_dim=144*k_num, out_dim=1280*k_num, **llkw))
        layers.append(LogicDense(in_dim=576*k_num, out_dim=1280*k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=1280*k_num, out_dim=640*k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=640*k_num, out_dim=320*k_num, parametrization=parametrization, **llkw))

        super(ClgnCifar10Tiny, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10Tiny32(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10Tiny32, self).__init__(k_num=32, tau=tau, **llkw)

class ClgnCifar10Tiny64(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10Tiny64, self).__init__(k_num=64, tau=tau, **llkw)

class ClgnCifar10Tiny128(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10Tiny128, self).__init__(k_num=128, tau=tau, **llkw)

class ClgnCifar10Tiny256(ClgnCifar10Tiny):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10Tiny256, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10Mini(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    Takes continuous (unthresholded) inputs. Only single conv layer.
    """
    conv_lut_rank = 2
    dense_lut_rank = 2
    tree_depth = 3
    n_input_bits = 3
    conv_parametrization = None
    dense_parametrization = None

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, 
                 k_num=16, tau=20, parametrization="raw", **llkw):
        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."

        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)  

        del llkw['lut_rank']  # remove lut_rank from llkw to avoid conflict

        assert (self.conv_parametrization is None) == (self.dense_parametrization is None), "conv and dense parametrization must be both None or both set."
        self._conv_parametrization = self.conv_parametrization if self.conv_parametrization is not None else parametrization
        self._dense_parametrization = self.dense_parametrization if self.dense_parametrization is not None else parametrization
        if 'parametrization' in llkw:
            del llkw['parametrization']  # remove parametrization from llkw to avoid conflict

        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=32,
                num_kernels=k_num,
                channels=3*n_bits,
                tree_depth=1,
                receptive_field_size=3,
                parametrization=parametrization,
                **llkw,
            )  # kx30x30=900k
        )
        layers.append(
            OrPooling2d(kernel_size=2, stride=2) # kx15x15=225k
        )

        layers.append(torch.nn.Flatten())

        # layers.append(LogicDense(in_dim=225*k_num, out_dim=160*k_num, parametrization=parametrization, **llkw))
        # layers.append(LogicDense(in_dim=160*k_num, out_dim=80*k_num, parametrization=parametrization, **llkw))

        layers.append(LogicDense(in_dim=225*k_num, out_dim=200*k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=200*k_num, out_dim=200*k_num, parametrization=parametrization, **llkw))

        super(ClgnCifar10Mini, self).__init__(*layers, GroupSum(k=10, tau=tau))



class ClgnCifar10Mini32(ClgnCifar10Mini):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 5)
        super(ClgnCifar10Mini32, self).__init__(k_num=32, tau=tau, **llkw)


class ClgnCifar10Mini32Mixed(ClgnCifar10Mini):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 5)
        self.conv_parametrization


class ClgnCifar10Mini64(ClgnCifar10Mini):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 10)
        super(ClgnCifar10Mini64, self).__init__(k_num=64, tau=tau, **llkw)

class ClgnCifar10Mini128(ClgnCifar10Mini):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 20)
        super(ClgnCifar10Mini128, self).__init__(k_num=128, tau=tau, **llkw)

class ClgnCifar10Mini256(ClgnCifar10Mini):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 40)
        super(ClgnCifar10Mini256, self).__init__(k_num=256, tau=tau, **llkw)


class ClgnCifar10TinyRes(torch.nn.Sequential):
    conv_lut_rank = 2
    dense_lut_rank = 2
    tree_depth = 3
    n_input_bits = 3
    conv_parametrization = None
    dense_parametrization = None

    # def __init__(self, n_bits: int, k_num: int, tau: float, parametrization="raw", **llkw):
    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, 
            k_num=32, tau=20, parametrization="raw", **llkw):

        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."

        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        del llkw['lut_rank']  # remove lut_rank from llkw to avoid conflict

        assert (self.conv_parametrization is None) == (self.dense_parametrization is None), "conv and dense parametrization must be both None or both set."
        self._conv_parametrization = self.conv_parametrization if self.conv_parametrization is not None else parametrization
        self._dense_parametrization = self.dense_parametrization if self.dense_parametrization is not None else parametrization
        if 'parametrization' in llkw:
            del llkw['parametrization']  # remove parametrization from llkw to avoid conflict
        
        layers = [binarization_module]
        layers.append(
            ResidualLogicBlockLiv(
                in_dim=32,
                in_channels=3*n_bits,
                out_channels=2*k_num,
                tree_depth=self.tree_depth,
                receptive_field_size=3,
                padding=1,
                downsample=True,
                lut_rank=self.conv_lut_rank,
                parametrization=self.conv_parametrization,
                **llkw,
            )
        )
        layers.append(torch.nn.Flatten()) # 4x4x16k = 256k

        layers.append(LogicDense(in_dim=256*2*k_num, out_dim=512*k_num, parametrization=self.dense_parametrization, lut_rank=self.dense_lut_rank, **llkw))
        layers.append(LogicDense(in_dim=512*k_num, out_dim=256*k_num, parametrization=self.dense_parametrization, lut_rank=self.dense_lut_rank, **llkw))
        layers.append(LogicDense(in_dim=256*k_num, out_dim=320*k_num, parametrization=self.dense_parametrization, lut_rank=self.dense_lut_rank, **llkw))

        for layer in layers:
            print(f"isinstance(layer, nn.Module): {isinstance(layer, nn.Module)}")

        super(ClgnCifar10TinyRes, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnCifar10TinyResMixed(ClgnCifar10TinyRes):
    conv_parametrization = "warp"
    dense_parametrization = "raw"
    conv_lut_rank = 4
    dense_lut_rank = 2
    tree_depth = 1
