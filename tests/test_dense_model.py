"""Test suite for the DLGN (Dense Logic Gate Network) implementation.
This module contains tests for a model that contains and dense layers.
"""

import pytest
import numpy as np
import torch

from torchlogix.models import Dlgn
from torchlogix import CompiledLogicNet
    

def test_dlgn_model():
    """Test the DLGN model with a simple input."""
    # Create a simple DLGN model
    model = Dlgn(
        in_dim=100,
        n_layers=3,
        neurons_per_layer=10,
        class_count=10,
        tau=1.0,
        device="cpu",
        connections='random',
        weight_init='random'
    )
    model.train(False)  # Switch model to eval mode

    # Create a dummy input tensor with 8 mnist-like images
    X = torch.randint(0, 2, (8, 1, 10, 10)).float()  # Shape: (batch_size, channels, height, width)

    # Get the model's prediction
    preds = model(X)
    
    # Check if the output is as expected (shape and type)
    assert preds.shape == (8, 10)  # Assuming class_count=10
    assert isinstance(preds, torch.Tensor)

    # Note: Model starts with Flatten, so input gets flattened to 100 before first LogicDense
    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 10, 10), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_dlgn_model.so", verbose=False)

    preds_compiled = compiled_model(X.bool().numpy())
    
    print(f"{preds.shape=}\n{preds_compiled.shape=}")

    print(f"preds =\n{preds}\npreds_compiled =\n{preds_compiled}")

    assert np.allclose(preds.numpy(), preds_compiled, atol=1e-5), "Compiled model predictions do not match original model predictions"

