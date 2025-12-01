

from abc import ABC, abstractmethod
import torch

from ..parametrization import RawLUTParametrization, WalshLUTParametrization


class LogicBase(torch.nn.Module, ABC):
    """
    Abstract base class for logic layers.
    Provides common functionality and enforces implementation of certain methods.
    """
    def __init__(
        self, 
        parametrization: str = "raw",
        device: str = "cuda",
        grad_factor: float = 1.0,
        temperature: float = 1.0,
        forward_sampling: str = "soft",
        lut_rank: int = 2,
        arbitrary_basis: bool = False,
        connections: str = "random",
        weight_init: str = "residual",
        residual_init_param: float = 1.0,
    ):
        super().__init__()
        # Create parametrization component (sampler merged into parametrization)
        if parametrization == "raw":
            self.parametrization = RawLUTParametrization(
                lut_rank, arbitrary_basis, forward_sampling, temperature
            )
        elif parametrization == "walsh":
            self.parametrization = WalshLUTParametrization(
                lut_rank, arbitrary_basis, forward_sampling, temperature
            )
        else:
            raise ValueError(
                f"Unsupported parametrization: {parametrization}. "
                f"Choose 'raw' or 'walsh'."
            )
        self.device = device
        self.grad_factor = grad_factor
        self.lut_rank = lut_rank
        self.connections = connections
        self.weight_init = weight_init
        self.residual_init_param = residual_init_param

    @abstractmethod
    def _init_weights(self, **kwargs):
        pass

    @abstractmethod
    def _init_connections(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def get_luts_and_ids(self, **kwargs):
        """Computes the most probable LUT and its ID for each neuron.

        Method is dependent on the chosen parametrization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``luts``: Boolean tensor of shape ``(out_dim, 2 ** lut_rank)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
                - ``ids``: Integer tensor of shape ``(out_dim,)`` where each
                  entry is the integer ID of the corresponding LUT, obtained by
                  interpreting its truth table as a binary number (or None if
                  not applicable for high lut_rank).
        """
        pass
    
    @abstractmethod
    def get_luts(self, **kwargs):
        """Computes the most probable LUT for each neuron.

        Method is dependent on the chosen parametrization.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(out_dim, 2 ** lut_rank)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
        """
        pass