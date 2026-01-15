import torch

from ..layers import GroupSum, LogicDense
from ..layers.binarization import setup_binarization


class Dlgn(torch.nn.Sequential):
    """
    Randomly connected logic gate network as described in the paper
    'Deep Differentiable Logic Gate Networks'.
    """
    n_input_bits = None  # optional, to be set in subclasses

    def __init__(
        self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict,
        in_dim: int, n_layers: int, neurons_per_layer: int, class_count: int, tau: float, **llkw
    ):
        if self.n_input_bits is not None:
            assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)
        layers = [binarization_module, torch.nn.Flatten()]
        layers.append(
            LogicDense(in_dim=in_dim, out_dim=neurons_per_layer, **llkw)
        )
        for _ in range(n_layers - 1):
            layers.append(
                LogicDense(in_dim=neurons_per_layer, out_dim=neurons_per_layer, **llkw)
            )
        super(Dlgn, self).__init__(*layers, GroupSum(class_count, tau))


class DlgnFashionMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the Fashion-MNIST dataset.
    """
    n_input_bits = 2

    def __init__(self, neurons_per_layer: int, tau: float, **llkw):
        llkw["binarization"] = "dummy"
        super(DlgnFashionMnist, self).__init__(
            in_dim=28*28,
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

class DlgnFashionMnistSmallRank4(DlgnFashionMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnFashionMnistSmallRank4, self).__init__(neurons_per_layer=4000, tau=tau, **llkw)

class DlgnFashionMnistSmallRank6(DlgnFashionMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnFashionMnistSmallRank6, self).__init__(neurons_per_layer=2333, tau=tau, **llkw)


class DlgnMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the MNIST dataset.
    """
    n_input_bits = 1  # All MNIST models use 1-bit inputs

    def __init__(self, neurons_per_layer: int, tau: float, **llkw):
        llkw["binarization"] = "dummy"
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

class DlgnMnistSmallRank4(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnMnistSmallRank4, self).__init__(neurons_per_layer=4000, tau=tau, **llkw)

class DlgnMnistSmallRank6(DlgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnMnistSmallRank6, self).__init__(neurons_per_layer=2333, tau=tau, **llkw)

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

    
class DlgnCifar10Deep(DlgnCifar10):
    n_input_bits = 3
    depth_factor = 1  # to be set in subclasses
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.01)
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
        super(DlgnJscMediumRank6, self).__init__(n_layers=4, neurons_per_layer=42_333, tau=tau, **llkw)

class DlgnJscMedium10BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 10

class DlgnJscMedium20BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 20

class DlgnJscMedium50BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 50

class DlgnJscMedium100BitsRank6(DlgnJscMediumRank6):
    n_input_bits = 100
