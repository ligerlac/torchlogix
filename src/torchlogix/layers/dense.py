import torch

from ..connections import setup_connections
from ..functional import (
    GradFactor, get_regularization_loss, rescale_weights
    )
from ..inference_state import set_persistent_buffer
from .base import LogicBase
from ..functional import apply_luts_vectorized_export_mode


class LogicDense(LogicBase):
    """Fully-connected logic gate layer with differentiable learning.

    This module provides the core implementation of Differentiable Deep Logic
    Gate Networks. Each neuron learns a Boolean logic function (LUT) that
    operates on a subset of input features.

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of neurons (output features).
        device (str): Device to run the layer on ('cpu' or 'cuda').
        grad_factor (float): Gradient scaling factor.
        lut_rank (int): Rank of the LUTs used in the layer.
        parametrization (str): Type of parametrization to use ('raw', 'warp', 'light').
        parametrization_kwargs (dict): Additional keyword arguments for parametrization.
        connections (str): Type of connections to use ('fixed', 'learnable', etc.).
        connections_kwargs (dict): Additional keyword arguments for connections.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        grad_factor: float = 1.0,
        lut_rank: int = 2,
        parametrization: str = "raw",
        parametrization_kwargs: dict = None,
        connections: str = "fixed",
        connections_kwargs: dict = None,
    ):
        super().__init__(
            device=device,
            grad_factor=grad_factor,
            lut_rank=lut_rank,
            parametrization=parametrization,
            parametrization_kwargs=parametrization_kwargs,
            connections=connections,
            connections_kwargs=connections_kwargs,
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self._init_weights(out_dim=out_dim)
        self.connections = self._init_connections()
        # Legacy attributes for compatibility
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def _init_weights(self, out_dim):
        # Initialize weights using parametrization
        weights = self.parametrization.init_weights(
            out_dim, self.device
        )
        return torch.nn.Parameter(weights)

    def forward(self, x):
        """Applies the LogicDense transformation to the input.

        For each neuron, the layer:
        1. Selects ``lut_rank`` input features according to the connection
           pattern in ``self.indices``.
        2. Samples (or selects) LUT weights based on ``self.weight`` and
           the sampler strategy.
        3. Evaluates the resulting binary operation.

        Args:
            x: Input tensor of shape ``(..., in_dim)``. The last dimension must
                match ``self.in_dim``.

        Returns:
            A tensor of shape ``(..., out_dim)`` containing the neuron outputs.
        """
        assert x.ndim >= 2, x.ndim
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        # Extract inputs according to connection pattern
        x = self.connections(x)  # Shape: (batch_size, lut_rank, out_dim)

        # Split into export and train path (optimized separately for efficiency and exportability)
        # Export path only needs to know which LUT, not how it's parameterized
        if self.export_mode:
            if self.lut_rank != 2:
                raise NotImplementedError("Export mode currently only supports lut_rank=2.")
            # TODO: apply_luts function w/ bit shifts that works on higher lut_ranks
            if isinstance(x, torch.Tensor):
                x = x.bool()
            else:
                x = x.astype(bool, copy=False)
            a, b = x[:, 0], x[:, 1]  # Assuming lut_rank=2 for export mode
            ids = self._export_lut_ids
            return apply_luts_vectorized_export_mode(a, b, ids)
        # Delegate to parametrization with einsum contraction
        # b=batch, n=neurons, k=num_basis
        return self.parametrization.forward(
            x, self.weight, self.training,
            contraction='n,bn->bn'
        )

    def extra_repr(self):
        weight_repr = f"Parameter containing: [{self.weight.dtype} of size {tuple(self.weight.shape)}"
        if self.weight.is_cuda:
            weight_repr += f" ({self.weight.device})"
        weight_repr += "]"
        return f"{self.in_dim}, {self.out_dim}\nweight: {weight_repr}\nparametrization: {self.parametrization.__class__.__name__}\nconnections: {self.connections.__class__.__name__}, {self.connections}"

    def _init_connections(self):
        """Constructs input–neuron connection indices."""
        self.connections = setup_connections(
            structure="dense",
            connections=self.connections,
            lut_rank=self.lut_rank,
            device=self.device,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            **self.connections_kwargs
        )
        return self.connections
        
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
        return self.parametrization.get_luts_and_ids(self.weight, **kwargs)
    
    def get_luts(self, **kwargs):
        """Computes the most probable LUT for each neuron.

        Method is dependent on the chosen parametrization.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(out_dim, 2 ** lut_rank)``,
                  where each row is the most probable LUT truth table for a
                  neuron (entry is True for output 1, False for 0).
        """
        return self.parametrization.get_luts(self.weight, **kwargs)

    def get_regularization_loss(self, regularizer: str):
        return get_regularization_loss(self.weight, regularizer)
    
    def rescale_weights(self, method):
        rescale_weights(self.weight, method)

    def _torchlogix_get_inference_state(self, prefix: str):
        if self.lut_rank != 2:
            raise NotImplementedError("Inference-only serialization currently only supports lut_rank=2.")

        _, lut_ids = self.get_luts_and_ids()
        state = {
            f"{prefix}_inference.lut_ids": lut_ids.detach().clone(),
        }

        if hasattr(self.connections, "weights"):
            state[f"{prefix}_inference.connection_candidates"] = self.connections.indices.detach().clone()
            state[f"{prefix}_inference.connection_selection"] = self.connections.weights.argmax(dim=0).detach().clone()
        else:
            state[f"{prefix}_inference.connection_indices"] = self.connections.indices.detach().clone()

        return state

    def _torchlogix_load_inference_state(self, state_dict, prefix: str, missing_keys, consumed_keys):
        lut_key = f"{prefix}_inference.lut_ids"
        if lut_key not in state_dict:
            missing_keys.append(lut_key)
            return

        lut_ids = state_dict[lut_key].to(device=self.weight.device, dtype=torch.int64)
        consumed_keys.add(lut_key)
        loaded_all_required_keys = True

        if hasattr(self.connections, "weights"):
            candidate_key = f"{prefix}_inference.connection_candidates"
            selection_key = f"{prefix}_inference.connection_selection"

            if candidate_key not in state_dict:
                missing_keys.append(candidate_key)
                loaded_all_required_keys = False
            if selection_key not in state_dict:
                missing_keys.append(selection_key)
                loaded_all_required_keys = False
            if candidate_key in state_dict and selection_key in state_dict:
                candidates = state_dict[candidate_key].to(device=self.weight.device, dtype=torch.int64)
                selection = state_dict[selection_key].to(device=self.weight.device, dtype=torch.int64)
                self.connections.indices = candidates
                new_weights = torch.zeros_like(self.connections.weights.data)
                src = torch.ones_like(selection, dtype=new_weights.dtype).unsqueeze(0)
                new_weights.scatter_(0, selection.unsqueeze(0), src)
                self.connections.weights.data.copy_(new_weights)
                consumed_keys.update({candidate_key, selection_key})
        else:
            conn_key = f"{prefix}_inference.connection_indices"
            if conn_key not in state_dict:
                missing_keys.append(conn_key)
                loaded_all_required_keys = False
            else:
                self.connections.indices = state_dict[conn_key].to(
                    device=self.weight.device,
                    dtype=torch.int64,
                )
                consumed_keys.add(conn_key)

        if not loaded_all_required_keys:
            return

        self._set_inference_only_mode()
        set_persistent_buffer(self, "_export_lut_ids", lut_ids)
