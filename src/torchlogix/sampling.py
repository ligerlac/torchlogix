"""Sampling strategies for logic gate networks.

This module provides different strategies for sampling discrete logic gates
from continuous weight distributions during training and evaluation.
"""

from abc import ABC, abstractmethod
import torch
from torch.nn.functional import gumbel_softmax

from .functional import softmax, sigmoid, gumbel_sigmoid


class Sampler(ABC):
    """Base class for weight sampling strategies.

    Samplers handle how to convert continuous weight parameters into
    discrete selections (during eval) or continuous relaxations (during training).
    """

    def __init__(self, forward_sampling: str, temperature: float = 1.0):
        """Initialize sampler.

        Args:
            forward_sampling: Sampling strategy - one of:
                - "soft": Continuous relaxation
                - "hard": Straight-through discrete selection
                - "gumbel_soft": Gumbel-Softmax continuous
                - "gumbel_hard": Gumbel-Softmax straight-through
            temperature: Temperature parameter for sampling.
        """
        self.forward_sampling = forward_sampling
        self.temperature = temperature

    @abstractmethod
    def sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        """Sample weights during training.

        Args:
            weights: Raw weight parameters.

        Returns:
            Sampled weights (continuous or straight-through).
        """
        pass

    @abstractmethod
    def sample_eval(self, weights: torch.Tensor, lut_entries: int = None) -> torch.Tensor:
        """Sample weights during evaluation.

        Args:
            weights: Raw weight parameters.
            lut_entries: Number of LUT entries (for one-hot encoding if needed).

        Returns:
            Discrete weight selection.
        """
        pass


class SoftmaxSampler(Sampler):
    """Sampler using softmax for raw LUT parametrization.

    Used with raw truth table parametrization where weights represent
    logits over all possible Boolean functions.
    """

    def __init__(self, forward_sampling: str, temperature: float = 1.0):
        super().__init__(forward_sampling, temperature)
        valid_modes = ["soft", "hard", "gumbel_soft", "gumbel_hard"]
        if forward_sampling not in valid_modes:
            raise ValueError(
                f"forward_sampling must be one of {valid_modes}, got {forward_sampling}"
            )

    def sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        print(f"sample_train: {weights.shape=}")
        print(f"sample_train: {weights=}")
        if self.forward_sampling == "soft":
            return softmax(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "hard":
            return softmax(weights, tau=self.temperature, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return gumbel_softmax(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "gumbel_hard":
            return gumbel_softmax(weights, tau=self.temperature, hard=True)

    def sample_eval(self, weights: torch.Tensor, lut_entries: int = None) -> torch.Tensor:
        """During eval, select the most probable LUT via argmax."""
        if lut_entries is None:
            raise ValueError("lut_entries required for SoftmaxSampler.sample_eval")
        return torch.nn.functional.one_hot(
            weights.argmax(-1), lut_entries
        ).to(torch.float32)


class SigmoidSampler(Sampler):
    """Sampler using sigmoid for Walsh parametrization.

    Used with Walsh-Hadamard parametrization where weights represent
    basis coefficients that are combined and passed through sigmoid.
    """

    def __init__(self, forward_sampling: str, temperature: float = 1.0):
        super().__init__(forward_sampling, temperature)
        valid_modes = ["soft", "hard", "gumbel_soft", "gumbel_hard"]
        if forward_sampling not in valid_modes:
            raise ValueError(
                f"forward_sampling must be one of {valid_modes}, got {forward_sampling}"
            )

    def sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        """Sample Walsh coefficients during training.

        Note: weights here are the combined Walsh coefficients (after basis
        multiplication), so we apply sigmoid/gumbel_sigmoid directly.
        """
        if self.forward_sampling == "soft":
            return sigmoid(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "hard":
            return sigmoid(weights, tau=self.temperature, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return gumbel_sigmoid(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "gumbel_hard":
            return gumbel_sigmoid(weights, tau=self.temperature, hard=True)

    def sample_eval(self, weights: torch.Tensor, lut_entries: int = None) -> torch.Tensor:
        """During eval, threshold at 0 for discrete output."""
        return (weights < 0).to(dtype=torch.float32)
