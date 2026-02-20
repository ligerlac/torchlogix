from typing import Union
import torch

from ..layers import GroupSum, LogicDense
from ..layers.binarization import setup_binarization


class Dlgn(torch.nn.Sequential):
    """
    Randomly connected logic gate network as described in the paper
    'Deep Differentiable Logic Gate Networks'.
    """
    n_input_bits = None  # optional, to be set in subclasses
    n_learnable_layers = 0

    def __init__(
        self,
        thresholds: torch.Tensor,
        binarization: str,
        binarization_kwargs: dict,
        in_dim: int,
        n_layers: int,
        neurons_per_layer: Union[int, list],
        class_count: int,
        tau: float,
        **llkw
    ):
        assert n_layers >= self.n_learnable_layers
        if isinstance(neurons_per_layer, int):
            neurons_per_layer = [neurons_per_layer] * n_layers
        assert len(neurons_per_layer) == n_layers, "Length of neurons_per_layer list must equal n_layers."
        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)
        layers = [binarization_module, torch.nn.Flatten()]
        if self.n_learnable_layers > 0:
            layers.append(
                LogicDense(in_dim=in_dim, out_dim=neurons_per_layer[0], **(llkw | {"connections": "learnable"}))
            )
        else:
            layers.append(
                LogicDense(in_dim=in_dim, out_dim=neurons_per_layer[0], **llkw)
            )
        for i in range(1, n_layers):
            if self.n_learnable_layers > i:
                layers.append(
                    LogicDense(in_dim=neurons_per_layer[i-1], out_dim=neurons_per_layer[i], **(llkw | {"connections": "learnable"}))
                )
            else:
                layers.append(
                    LogicDense(in_dim=neurons_per_layer[i-1], out_dim=neurons_per_layer[i], **llkw)
                )
        super(Dlgn, self).__init__(*layers, GroupSum(class_count, tau))


class DlgnFashionMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the Fashion-MNIST dataset.
    """
    n_input_bits = 3  # All Fashion-MNIST models use 3-bit inputs
    n_learnable_layers = 0

    def __init__(self, neurons_per_layer: int, tau: float, **llkw):
        llkw["binarization"] = "uniform"
        super(DlgnFashionMnist, self).__init__(
            in_dim=28*28*self.n_input_bits,
            n_layers=5,
            neurons_per_layer=neurons_per_layer,
            class_count=10,
            tau=tau,
            **llkw
        )


class DlgnFashionMnistSmall(DlgnFashionMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnFashionMnistSmall, self).__init__(neurons_per_layer=8000, tau=tau, **llkw)

class DlgnFashionMnistSmallLearn1(DlgnFashionMnistSmall):
    n_learnable_layers = 1

class DlgnFashionMnistSmallRank4(DlgnFashionMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnFashionMnistSmallRank4, self).__init__(neurons_per_layer=4000, tau=tau, **llkw)

class DlgnFashionMnistSmallRank4Learn1(DlgnFashionMnistSmallRank4):
    n_learnable_layers = 1

class DlgnFashionMnistSmallRank6(DlgnFashionMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnFashionMnistSmallRank6, self).__init__(neurons_per_layer=2330, tau=tau, **llkw)

class DlgnFashionMnistSmallRank6Learn1(DlgnFashionMnistSmallRank6):
    n_learnable_layers = 1

class DlgnMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the MNIST dataset.
    """
    n_input_bits = 1  # All MNIST models use 1-bit inputs
    n_learnable_layers = 0

    def __init__(self, neurons_per_layer: int, tau: float, **llkw):
        super(DlgnMnist, self).__init__(
            in_dim=28*28,
            n_layers=5,
            neurons_per_layer=neurons_per_layer,
            class_count=10,
            tau=tau,
            **llkw
        )

class DlgnMnistTiny(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnMnistTiny, self).__init__(neurons_per_layer=1000, tau=tau, **llkw)

class DlgnMnistSmall(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnMnistSmall, self).__init__(neurons_per_layer=8000, tau=tau, **llkw)

class DlgnMnistSmallLearn1(DlgnMnistSmall):
    n_learnable_layers = 1

class DlgnMnistSmallRank4(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnMnistSmallRank4, self).__init__(neurons_per_layer=4000, tau=tau, **llkw)

class DlgnMnistSmallRank4Learn1(DlgnMnistSmallRank4):
    n_learnable_layers = 1

class DlgnMnistSmallRank6(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnMnistSmallRank6, self).__init__(neurons_per_layer=2330, tau=tau, **llkw)

class DlgnMnistSmallRank6Learn1(DlgnMnistSmallRank6):
    n_learnable_layers = 1

class DlgnMnistMedium(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnMnistMedium, self).__init__(neurons_per_layer=64000, tau=tau, **llkw)


class DlgnCifar10(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the CIFAR-10 dataset.
    Using 3 color channels and 3-bit-per-channel encoding.
    """
    n_learnable_layers = 0

    def __init__(self, n_layers: int, neurons_per_layer: int, tau: float, **llkw):
        n_bits = llkw["thresholds"].shape[-1]
        llkw["binarization_kwargs"]["feature_dim"] = 1
        super(DlgnCifar10, self).__init__(
            in_dim=3*32*32*n_bits,
            n_layers=n_layers,
            neurons_per_layer=neurons_per_layer,
            class_count=10,
            tau=tau,
            **llkw
        )

    
class DlgnCifar10MediumDeep(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10MediumDeep, self).__init__(
            n_layers=4*self.depth_factor, neurons_per_layer=128_000, tau=tau, **llkw
        )

class DlgnCifar10MediumDeepRank4(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10MediumDeepRank4, self).__init__(
            n_layers=4*self.depth_factor, neurons_per_layer=56_000, tau=tau, **llkw
        )

class DlgnCifar10MediumDeepRank6(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10MediumDeepRank6, self).__init__(
            n_layers=4*self.depth_factor, neurons_per_layer=42_000, tau=tau, **llkw
        )

class DlgnCifar10MediumDeep3(DlgnCifar10MediumDeep):
    depth_factor = 3

class DlgnCifar10MediumDeep3Rank4(DlgnCifar10MediumDeepRank4):
    depth_factor = 3

class DlgnCifar10MediumDeep3Rank6(DlgnCifar10MediumDeepRank6):
    depth_factor = 3


class DlgnCifar10Deep(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10Deep, self).__init__(
            n_layers=4*self.depth_factor, neurons_per_layer=12_000, tau=tau, **llkw
        )

class DlgnCifar10Deep2(DlgnCifar10Deep):
    depth_factor = 2

class DlgnCifar10Deep3(DlgnCifar10Deep):
    depth_factor = 3

class DlgnCifar10Deep4(DlgnCifar10Deep):
    depth_factor = 4

class DlgnCifar10Deep5(DlgnCifar10Deep):
    depth_factor = 5

class DlgnCifar10Deep10(DlgnCifar10Deep):
    depth_factor = 10

class DlgnCifar10Deep20(DlgnCifar10Deep):
    depth_factor = 20


class DlgnCifar10DeepRank4(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10DeepRank4, self).__init__(
            n_layers=4*self.depth_factor, neurons_per_layer=6_000, tau=tau, **llkw
        )

class DlgnCifar10Deep2Rank4(DlgnCifar10DeepRank4):
    depth_factor = 2

class DlgnCifar10Deep3Rank4(DlgnCifar10DeepRank4):
    depth_factor = 3

class DlgnCifar10Deep4Rank4(DlgnCifar10DeepRank4):
    depth_factor = 4

class DlgnCifar10Deep5Rank4(DlgnCifar10DeepRank4):
    depth_factor = 5

class DlgnCifar10Deep10Rank4(DlgnCifar10DeepRank4):
    depth_factor = 10

class DlgnCifar10Deep20Rank4(DlgnCifar10DeepRank4):
    depth_factor = 20


class DlgnCifar10DeepRank6(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10DeepRank6, self).__init__(
            n_layers=4*self.depth_factor, neurons_per_layer=4000, tau=tau, **llkw
        )

class DlgnCifar10Deep2Rank6(DlgnCifar10DeepRank6):
    depth_factor = 2

class DlgnCifar10Deep3Rank6(DlgnCifar10DeepRank6):
    depth_factor = 3

class DlgnCifar10Deep4Rank6(DlgnCifar10DeepRank6):
    depth_factor = 4

class DlgnCifar10Deep5Rank6(DlgnCifar10DeepRank6):
    depth_factor = 5

class DlgnCifar10Deep10Rank6(DlgnCifar10DeepRank6):
    depth_factor = 10

class DlgnCifar10Deep20Rank6(DlgnCifar10DeepRank6):
    depth_factor = 20



class DlgnCifar10Small(DlgnCifar10):
    n_input_bits = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10Small, self).__init__(
            n_layers=4, neurons_per_layer=12_000, tau=tau, **llkw
        )


class DlgnCifar10SmallRank4(DlgnCifar10):
    n_input_bits = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10SmallRank4, self).__init__(
            n_layers=4, neurons_per_layer=6_000, tau=tau, **llkw
        )


class DlgnCifar10SmallRank6(DlgnCifar10):
    n_input_bits = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10SmallRank6, self).__init__(
            n_layers=4, neurons_per_layer=3000, tau=tau, **llkw
        )


class DlgnCifar10Medium(DlgnCifar10):
    n_input_bits = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10Medium, self).__init__(
            n_layers=4, neurons_per_layer=128_000, tau=tau, **llkw
        )


class DlgnCifar10Large(DlgnCifar10):
    n_input_bits = 5
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10Large, self).__init__(
            n_layers=5, neurons_per_layer=256_000, tau=tau, **llkw
        )


class DlgnCifar10Large2(DlgnCifar10):
    n_input_bits = 5
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10Large2, self).__init__(
            n_layers=5, neurons_per_layer=512_000, tau=tau, **llkw
        )


class DlgnCifar10Large4(DlgnCifar10):
    n_input_bits = 5
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
        super(DlgnCifar10Large4, self).__init__(
            n_layers=5, neurons_per_layer=1_024_000, tau=tau, **llkw
        )


class DlgnJsc(Dlgn):
    """
    Model as described in the paper 'LLNN'
    """
    n_input_bits = None  # to be set in subclasses
    n_learnable_layers = 0
    def __init__(self, n_layers: int, neurons_per_layer: int, tau: float, **llkw):
        n_bits = llkw["thresholds"].shape[-1]
        llkw["binarization_kwargs"]["feature_dim"] = 1
        super(DlgnJsc, self).__init__(
            in_dim=16*n_bits,
            n_layers=n_layers,
            neurons_per_layer=neurons_per_layer,
            class_count=5,
            tau=tau,
            **llkw
        )


class DlgnJscSmall(DlgnJsc):
    n_input_bits = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.02)
        super(DlgnJscSmall, self).__init__(n_layers=2, neurons_per_layer=32_000, tau=tau, **llkw)


class DlgnJscSmall10Bits(DlgnJscSmall):
    n_input_bits = 10

class DlgnJscSmall20Bits(DlgnJscSmall):
    n_input_bits = 20

class DlgnJscSmall50Bits(DlgnJscSmall):
    n_input_bits = 50

class DlgnJscSmall100Bits(DlgnJscSmall):
    n_input_bits = 100


class DlgnJscMedium(DlgnJsc):
    n_input_bits = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.02)
        super(DlgnJscMedium, self).__init__(n_layers=4, neurons_per_layer=128_000, tau=tau, **llkw)

class DlgnJscMedium10Bits(DlgnJscMedium):
    n_input_bits = 10

class DlgnJscMedium20Bits(DlgnJscMedium):
    n_input_bits = 20

class DlgnJscMedium50Bits(DlgnJscMedium):
    n_input_bits = 50

class DlgnJscMedium100Bits(DlgnJscMedium):
    n_input_bits = 100

class DlgnJscSmallRank4(DlgnJsc):
    n_input_bits = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.02)
        super(DlgnJscSmallRank4, self).__init__(n_layers=2, neurons_per_layer=16_000, tau=tau, **llkw)

class DlgnJscSmall10BitsRank4(DlgnJscSmallRank4):
    n_input_bits = 10

class DlgnJscSmall20BitsRank4(DlgnJscSmallRank4):
    n_input_bits = 20

class DlgnJscSmall50BitsRank4(DlgnJscSmallRank4):
    n_input_bits = 50

class DlgnJscSmall100BitsRank4(DlgnJscSmallRank4):
    n_input_bits = 100

class DlgnJscSmallRank6(DlgnJsc):
    n_input_bits = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.02)
        super(DlgnJscSmallRank6, self).__init__(n_layers=2, neurons_per_layer=10_666, tau=tau, **llkw)

class DlgnJscSmall10BitsRank6(DlgnJscSmallRank6):
    n_input_bits = 10

class DlgnJscSmall20BitsRank6(DlgnJscSmallRank6):
    n_input_bits = 20

class DlgnJscSmall50BitsRank6(DlgnJscSmallRank6):
    n_input_bits = 50

class DlgnJscSmall100BitsRank6(DlgnJscSmallRank6):
    n_input_bits = 100

class DlgnJscMediumRank4(DlgnJsc):
    n_input_bits = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.02)
        super(DlgnJscMediumRank4, self).__init__(n_layers=4, neurons_per_layer=64_000, tau=tau, **llkw)

class DlgnJscMedium10BitsRank4(DlgnJscMediumRank4):
    n_input_bits = 10

class DlgnJscMedium20BitsRank4(DlgnJscMediumRank4):
    n_input_bits = 20

class DlgnJscMedium50BitsRank4(DlgnJscMediumRank4):
    n_input_bits = 50

class DlgnJscMedium100BitsRank4(DlgnJscMediumRank4):
    n_input_bits = 100

class DlgnJscMediumRank6(DlgnJsc):
    n_input_bits = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.02)
        super(DlgnJscMediumRank6, self).__init__(n_layers=4, neurons_per_layer=42_330, tau=tau, **llkw)

class DlgnJscMedium10BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 10

class DlgnJscMedium20BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 20

class DlgnJscMedium50BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 50

class DlgnJscMedium100BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 100


############################## DWN ##################################
class DlgnMnistDwn(Dlgn):
    """
    Model as described in the paper
    'Deep Weightless Networks'
    for the MNIST dataset.
    """
    n_input_bits = 3
    n_learnable_layers = None

    def __init__(self, neurons_per_layer: list, tau: float, **llkw):
        llkw["binarization"] = "distributive"
        super(DlgnMnistDwn, self).__init__(
            in_dim=28*28*self.n_input_bits,
            n_layers=len(neurons_per_layer),
            neurons_per_layer=neurons_per_layer,
            class_count=10,
            tau=tau,
            **llkw
        )

class DlgnMnistDwnLargeRank2(DlgnMnistDwn):
    n_learnable_layers = None

    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.071)
        super(DlgnMnistDwnLargeRank2, self).__init__(
            neurons_per_layer=[6_000, 6_000], 
            tau=tau, 
            **llkw)
        
class DlgnMnistDwnLargeRank2Learn0(DlgnMnistDwnLargeRank2):
    n_learnable_layers = 0

class DlgnMnistDwnLargeRank2Learn1(DlgnMnistDwnLargeRank2):
    n_learnable_layers = 1

class DlgnMnistDwnLargeRank2Learn2(DlgnMnistDwnLargeRank2):
    n_learnable_layers = 2
        
class DlgnMnistDwnSmallRank6(DlgnMnistDwn):
    n_learnable_layers = None
    n_input_bits = 1

    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.245)
        super(DlgnMnistDwnSmallRank6, self).__init__(
            neurons_per_layer=[1_000, 500], 
            tau=tau, 
            **llkw)
        
class DlgnMnistDwnSmallRank6Learn0(DlgnMnistDwnSmallRank6):
    n_learnable_layers = 0

class DlgnMnistDwnSmallRank6Learn1(DlgnMnistDwnSmallRank6):
    n_learnable_layers = 1

class DlgnMnistDwnSmallRank6Learn2(DlgnMnistDwnSmallRank6):
    n_learnable_layers = 2
        
class DlgnMnistDwnLargeRank6(DlgnMnistDwn):
    n_learnable_layers = None

    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.173)
        super(DlgnMnistDwnLargeRank6, self).__init__(
            neurons_per_layer=[2_000, 1_000], 
            tau=tau, 
            **llkw)
        
class DlgnMnistDwnLargeRank6Learn0(DlgnMnistDwnLargeRank6):
    n_learnable_layers = 0

class DlgnMnistDwnLargeRank6Learn1(DlgnMnistDwnLargeRank6):
    n_learnable_layers = 1

class DlgnMnistDwnLargeRank6Learn2(DlgnMnistDwnLargeRank6):
    n_learnable_layers = 2
        

class DlgnCifar10Dwn(Dlgn):
    """
    Model as described in the paper 'Differentiable Weightless Neural Networks' 
    for the Cifar-10 dataset.
    """
    n_learnable_layers = 0
    n_input_bits = 10

    def __init__(self, neurons_per_layer: list, tau: float, **llkw):
        llkw["binarization_kwargs"]["feature_dim"] = 1
        super(DlgnCifar10Dwn, self).__init__(
            in_dim=3*32*32*self.n_input_bits,
            n_layers=len(neurons_per_layer),
            neurons_per_layer=neurons_per_layer,
            class_count=10,
            tau=tau,
            **llkw
        )

class DlgnCifar10DwnRank2(DlgnCifar10Dwn):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10DwnRank2, self).__init__(
            neurons_per_layer=[24_000, 24_000], 
            tau=tau, 
            **llkw)
        
class DlgnCifar10DwnRank2Learn2(DlgnCifar10DwnRank2):
    n_learnable_layers = 2

class DlgnCifar10DwnRank6(DlgnCifar10Dwn):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10DwnRank6, self).__init__(
            neurons_per_layer=[8_000], 
            tau=tau, 
            **llkw)
        


class DlgnFashionMnistDwn(Dlgn):
    """
    Model as described in the paper 'Differentiable Weightless Neural Networks' 
    for the Fashion-MNIST dataset.
    """
    n_learnable_layers = 0
    n_input_bits = 7

    def __init__(self, neurons_per_layer: list, tau: float, **llkw):
        super(DlgnFashionMnistDwn, self).__init__(
            in_dim=28*28*self.n_input_bits,
            n_layers=len(neurons_per_layer),
            neurons_per_layer=neurons_per_layer,
            class_count=10,
            tau=tau,
            **llkw
        )

class DlgnFashionMnistDwnRank2(DlgnFashionMnistDwn):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.061)
        super(DlgnFashionMnistDwnRank2, self).__init__(
            neurons_per_layer=[8_000, 8_000], 
            tau=tau, 
            **llkw)
        
class DlgnFashionMnistDwnRank2Learn2(DlgnFashionMnistDwnRank2):
    n_learnable_layers = 2

class DlgnFashionMnistDwnRank6(DlgnFashionMnistDwn):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.122)
        super(DlgnFashionMnistDwnRank6, self).__init__(
            neurons_per_layer=[2_000, 2_000], 
            tau=tau, 
            **llkw)
        
class DlgnFashionMnistDwnRank6Learn2(DlgnFashionMnistDwnRank6):
    n_learnable_layers = 2


class DlgnJscDwn(Dlgn):
    """
    Model as described in the paper 'Differentiable Weightless Neural Networks' 
    for the JSC dataset.
    """
    n_learnable_layers = None
    n_input_bits = 200

    def __init__(self, neurons_per_layer: list, tau: float, **llkw):
        super(DlgnJscDwn, self).__init__(
            in_dim=16*self.n_input_bits,
            n_layers=len(neurons_per_layer),
            neurons_per_layer=neurons_per_layer,
            class_count=5,
            tau=tau,
            **llkw
        )


class DlgnJscDwnTinyRank6(DlgnJscDwn):
    n_learnable_layers = None

    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.7)
        super(DlgnJscDwnTinyRank6, self).__init__(
            neurons_per_layer=[10], 
            tau=tau, 
            **llkw)
        
class DlgnJscDwnTinyRank6Learn0(DlgnJscDwnTinyRank6):
    n_learnable_layers = 0

class DlgnJscDwnTinyRank6Learn1(DlgnJscDwnTinyRank6):
    n_learnable_layers = 1

class DlgnJscDwnSmallRank6(DlgnJscDwn):
    n_learnable_layers = None

    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.3)
        super(DlgnJscDwnSmallRank6, self).__init__(
            neurons_per_layer=[50], 
            tau=tau, 
            **llkw)


class DlgnJscDwnSmallRank6Bits1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 1

class DlgnJscDwnSmallRank6Bits2(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 2

class DlgnJscDwnSmallRank6Bits5(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 5

class DlgnJscDwnSmallRank6Bits10(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 10

class DlgnJscDwnSmallRank6Bits20(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 20

class DlgnJscDwnSmallRank6Bits50(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 50

class DlgnJscDwnSmallRank6Bits100(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 100

class DlgnJscDwnSmallRank6Bits200(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0
    n_input_bits = 200      


class DlgnJscDwnSmallRank6Bits1Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 1

class DlgnJscDwnSmallRank6Bits2Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 2

class DlgnJscDwnSmallRank6Bits5Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 5

class DlgnJscDwnSmallRank6Bits10Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 10

class DlgnJscDwnSmallRank6Bits20Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 20

class DlgnJscDwnSmallRank6Bits50Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 50

class DlgnJscDwnSmallRank6Bits100Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 100

class DlgnJscDwnSmallRank6Bits200Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1
    n_input_bits = 200

        
class DlgnJscDwnSmallRank6Learn0(DlgnJscDwnSmallRank6):
    n_learnable_layers = 0

class DlgnJscDwnSmallRank6Learn1(DlgnJscDwnSmallRank6):
    n_learnable_layers = 1


class DlgnJscDwnMediumRank2(DlgnJscDwn):
    n_learnable_layers = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnJscDwnMediumRank2, self).__init__(
            neurons_per_layer=[1080], 
            tau=tau, 
            **llkw)
        
class DlgnJscDwnMediumRank2Bits2(DlgnJscDwnMediumRank2):
    n_input_bits = 2
    n_learnable_layers = 0

class DlgnJscDwnMediumRank2Bits5(DlgnJscDwnMediumRank2):
    n_input_bits = 5
    n_learnable_layers = 0

class DlgnJscDwnMediumRank2Bits10(DlgnJscDwnMediumRank2):
    n_input_bits = 10
    n_learnable_layers = 0

class DlgnJscDwnMediumRank2Bits20(DlgnJscDwnMediumRank2):
    n_input_bits = 20
    n_learnable_layers = 0

class DlgnJscDwnMediumRank2Bits50(DlgnJscDwnMediumRank2):
    n_input_bits = 50
    n_learnable_layers = 0

class DlgnJscDwnMediumRank2Bits100(DlgnJscDwnMediumRank2):
    n_input_bits = 100
    n_learnable_layers = 0

        
class DlgnJscDwnMediumRank4(DlgnJscDwn):
    n_learnable_layers = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnJscDwnMediumRank4, self).__init__(
            neurons_per_layer=[540], 
            tau=tau, 
            **llkw)

class DlgnJscDwnMediumRank4Bits2(DlgnJscDwnMediumRank4):
    n_input_bits = 2
    n_learnable_layers = 0

class DlgnJscDwnMediumRank4Bits5(DlgnJscDwnMediumRank4):
    n_input_bits = 5
    n_learnable_layers = 0

class DlgnJscDwnMediumRank4Bits10(DlgnJscDwnMediumRank4):
    n_input_bits = 10
    n_learnable_layers = 0

class DlgnJscDwnMediumRank4Bits20(DlgnJscDwnMediumRank4):
    n_input_bits = 20
    n_learnable_layers = 0

class DlgnJscDwnMediumRank4Bits50(DlgnJscDwnMediumRank4):
    n_input_bits = 50
    n_learnable_layers = 0

class DlgnJscDwnMediumRank4Bits100(DlgnJscDwnMediumRank4):
    n_input_bits = 100
    n_learnable_layers = 0

        
class DlgnJscDwnMediumRank6(DlgnJscDwn):
    n_learnable_layers = None
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnJscDwnMediumRank6, self).__init__(
            neurons_per_layer=[360], 
            tau=tau, 
            **llkw)

class DlgnJscDwnMediumRank6Bits2(DlgnJscDwnMediumRank6):
    n_input_bits = 2
    n_learnable_layers = 0

class DlgnJscDwnMediumRank6Bits5(DlgnJscDwnMediumRank6):
    n_input_bits = 5
    n_learnable_layers = 0

class DlgnJscDwnMediumRank6Bits10(DlgnJscDwnMediumRank6):
    n_input_bits = 10
    n_learnable_layers = 0

class DlgnJscDwnMediumRank6Bits20(DlgnJscDwnMediumRank6):
    n_input_bits = 20
    n_learnable_layers = 0

class DlgnJscDwnMediumRank6Bits50(DlgnJscDwnMediumRank6):
    n_input_bits = 50
    n_learnable_layers = 0

class DlgnJscDwnMediumRank6Bits100(DlgnJscDwnMediumRank6):
    n_input_bits = 100
    n_learnable_layers = 0



class DlgnJscDwnMediumRank6Learn0(DlgnJscDwnMediumRank6):
    n_learnable_layers = 0

class DlgnJscDwnMediumRank6Learn1(DlgnJscDwnMediumRank6):
    n_learnable_layers = 1
        
class DlgnJscDwnLargeRank6(DlgnJscDwn):
    n_learnable_layers = None

    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnJscDwnLargeRank6, self).__init__(
            neurons_per_layer=[2400], 
            tau=tau,
            **llkw)

class DlgnJscDwnLargeRank6Learn0(DlgnJscDwnLargeRank6):
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Learn1(DlgnJscDwnLargeRank6):
    n_learnable_layers = 1

class DlgnJscDwnLargeRank6Bits2(DlgnJscDwnLargeRank6):
    n_input_bits = 2
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Bits5(DlgnJscDwnLargeRank6):
    n_input_bits = 5
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Bits10(DlgnJscDwnLargeRank6):
    n_input_bits = 10
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Bits20(DlgnJscDwnLargeRank6):
    n_input_bits = 20
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Bits50(DlgnJscDwnLargeRank6):
    n_input_bits = 50
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Bits100(DlgnJscDwnLargeRank6):
    n_input_bits = 100
    n_learnable_layers = 0

class DlgnJscDwnLargeRank6Bits200(DlgnJscDwnLargeRank6):
    n_input_bits = 200
    n_learnable_layers = 0
