from typing import Union
import torch

from torchlogix.layers.groupsum import setup_group_sum

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
        group_sum_method: str,
        group_sum_kwargs: dict,
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
        super(Dlgn, self).__init__(*layers, setup_group_sum(group_sum_method, **group_sum_kwargs))
        

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
            group_sum_method="groupsum",
            group_sum_kwargs={"k": 10, "tau": tau},
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

class DlgnAffineMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the MNIST dataset.
    """
    n_input_bits = 1  # All MNIST models use 1-bit inputs
    n_learnable_layers = 0

    def __init__(self, neurons_per_layer: int, **llkw):
        super(DlgnAffineMnist, self).__init__(
            in_dim=28*28,
            n_layers=5,
            neurons_per_layer=neurons_per_layer,
            group_sum_method="learnable_affine",
            group_sum_kwargs={"k": 10},
            **llkw
        )

class DlgnAffineMnistTiny(DlgnAffineMnist):
    def __init__(self, **llkw):
        super(DlgnAffineMnistTiny, self).__init__(neurons_per_layer=1000, **llkw)

class DlgnAffineMnistSmall(DlgnAffineMnist):
    def __init__(self, **llkw):
        super(DlgnAffineMnistSmall, self).__init__(neurons_per_layer=8000, **llkw)

class DlgnAffineMnistMedium(DlgnAffineMnist):
    def __init__(self, **llkw):
        super(DlgnAffineMnistMedium, self).__init__(neurons_per_layer=64000, **llkw)

class DlgnRegMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the MNIST dataset.
    """
    n_input_bits = 1  # All MNIST models use 1-bit inputs
    n_learnable_layers = 0

    def __init__(self, neurons_per_layer: int, tau: float, **llkw):
        super(DlgnRegMnist, self).__init__(
            in_dim=28*28,
            n_layers=5,
            neurons_per_layer=neurons_per_layer,
            group_sum_method="groupsum",
            group_sum_kwargs={"k": 1, "tau": tau},
            **llkw
        )

class DlgnRegMnistTiny(DlgnRegMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnRegMnistTiny, self).__init__(neurons_per_layer=1000, tau=tau, **llkw)

class DlgnRegMnistSmall(DlgnRegMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.1)
        super(DlgnRegMnistSmall, self).__init__(neurons_per_layer=8000, tau=tau, **llkw)

class DlgnRegMnistMedium(DlgnRegMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnRegMnistMedium, self).__init__(neurons_per_layer=64000, tau=tau, **llkw)

class DlgnRegAffineMnist(Dlgn):
    """
    Model as described in the paper
    'Deep Differentiable Logic Gate Networks'
    for the MNIST dataset.
    """
    n_input_bits = 1  # All MNIST models use 1-bit inputs
    n_learnable_layers = 0

    def __init__(self, neurons_per_layer: int, **llkw):
        super(DlgnRegAffineMnist, self).__init__(
            in_dim=28*28,
            n_layers=5,
            neurons_per_layer=neurons_per_layer,
            group_sum_method="learnable_affine",
            group_sum_kwargs={"k": 1},
            **llkw
        )

class DlgnRegAffineMnistTiny(DlgnRegAffineMnist):
    def __init__(self, **llkw):
        super(DlgnRegAffineMnistTiny, self).__init__(neurons_per_layer=1000, **llkw)

class DlgnRegAffineMnistSmall(DlgnRegAffineMnist):
    def __init__(self, **llkw):
        super(DlgnRegAffineMnistSmall, self).__init__(neurons_per_layer=8000, **llkw)

class DlgnRegAffineMnistMedium(DlgnRegAffineMnist):
    def __init__(self, **llkw):
        super(DlgnRegAffineMnistMedium, self).__init__(neurons_per_layer=64000, **llkw)



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

class DlgnCifar10Small(DlgnCifar10):
    n_input_bits = 3
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1./0.03)
        super(DlgnCifar10Small, self).__init__(
            n_layers=4, neurons_per_layer=12_000, tau=tau, **llkw
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
