

from abc import ABC, abstractmethod
import torch

from ..parametrization import RawLUTParametrization, WalshLUTParametrization, LightLUTParametrization, setup_parametrization


class LogicBase(torch.nn.Module, ABC):
    """
    Abstract base class for logic layers.
    Provides common functionality and enforces implementation of certain methods.

    Args:
        device (str): Device to run the layer on ('cpu' or 'cuda').
        grad_factor (float): Gradient scaling factor.
        lut_rank (int): Rank of the LUTs used in the layer.
        parametrization (str): Type of parametrization to use ('raw', 'walsh', 'light').
        parametrization_kwargs (dict): Additional keyword arguments for parametrization.
        connections (str): Type of connections to use ('fixed', 'learnable', etc.).
        connections_kwargs (dict): Additional keyword arguments for connections.
    """
    def __init__(
        self, 
        device: str = "cuda",
        grad_factor: float = 1.0,
        lut_rank: int = 2,
        parametrization: str = "raw",
        parametrization_kwargs: dict = None,
        connections: str = "fixed",
        connections_kwargs: dict = None,    
    ):
        super().__init__()
        # Create parametrization component (sampler merged into parametrization)
        self.parametrization = setup_parametrization(
            parametrization,
            lut_rank,
            **(parametrization_kwargs or {})
        )
        self.device = device
        self.grad_factor = grad_factor
        self.lut_rank = lut_rank
        self.connections = connections
        self.connections_kwargs = connections_kwargs or {}

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

    def get_regularization_loss(self, regularizer: str):
        """Computes regularization loss for the layer.

        Returns:
            torch.Tensor: Scalar tensor representing the regularization loss.
        """
        pass

    def rescale_weights(self, method: str):
        """Rescales the weights of the layer according to the specified method.

        Args:
            method (str): Rescaling method. Options are 'clip', 'abs_sum', 'L2'.
        """
        pass