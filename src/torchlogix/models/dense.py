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


class DlgnCifar10Small(DlgnCifar10):
    n_input_bits = 2
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10Small, self).__init__(
            n_layers=4, neurons_per_layer=12_000, tau=tau, **llkw
        )


class DlgnCifar10Medium(DlgnCifar10):
    n_input_bits = 2
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
