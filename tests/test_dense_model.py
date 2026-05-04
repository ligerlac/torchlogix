"""Test suite for the DLGN (Dense Logic Gate Network) implementation.
This module contains tests for a model that contains and dense layers.
"""

import time

import numpy as np
import torch

from torchlogix.models import Dlgn
from torchlogix import CompiledLogicNet


def _median_runtime(fn, warmup=3, repeats=7):
    for _ in range(warmup):
        fn()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)

    return float(np.median(timings))
    

def test_dlgn_model():
    """Test the DLGN model with a simple input."""
    # Create a simple DLGN model
    model = Dlgn(
        thresholds=None,
        binarization="dummy",
        binarization_kwargs={},
        in_dim=100,
        n_layers=3,
        neurons_per_layer=50,
        group_sum_method="groupsum",
        group_sum_kwargs={"k": 10, "tau": 1.0},
        device="cpu",
        connections_kwargs={"init_method": "random"},
        parametrization_kwargs={"weight_init": "random"},
        parametrization='raw'
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


def test_dlgn_affine_model():
    """Test the DLGN model with a simple input."""
    # Create a simple DLGN model
    model = Dlgn(
        thresholds=None,
        binarization="dummy",
        binarization_kwargs={},
        in_dim=100,
        n_layers=3,
        neurons_per_layer=50,
        group_sum_method="learnable_affine",
        group_sum_kwargs={"k": 10},
        device="cpu",
        connections_kwargs={"init_method": "random"},
        parametrization_kwargs={"weight_init": "random"},
        parametrization='raw'
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

    assert np.allclose(preds.detach().numpy(), preds_compiled, atol=1e-5), "Compiled model predictions do not match original model predictions"


def test_dlgn_linear_model():
    """Test compiled DLGN inference with a LearnableGroupLinear output layer."""
    model = Dlgn(
        thresholds=None,
        binarization="dummy",
        binarization_kwargs={},
        in_dim=100,
        n_layers=3,
        neurons_per_layer=50,
        group_sum_method="learnable_linear",
        group_sum_kwargs={"k": 10},
        device="cpu",
        connections_kwargs={"init_method": "random"},
        parametrization_kwargs={"weight_init": "random"},
        parametrization='raw'
    )
    model.train(False)

    X = torch.randint(0, 2, (8, 1, 10, 10)).float()
    preds = model(X)

    assert preds.shape == (8, 10)
    assert isinstance(preds, torch.Tensor)

    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 10, 10), num_bits=8, cpu_compiler="gcc", verbose=True
    )
    compiled_model.compile(save_lib_path="compiled_dlgn_model.so", verbose=False)

    preds_compiled = compiled_model(X.bool().numpy())

    print(f"{preds.shape=}\n{preds_compiled.shape=}")
    print(f"preds =\n{preds}\npreds_compiled =\n{preds_compiled}")

    assert np.allclose(preds.detach().numpy(), preds_compiled, atol=1e-5), "Compiled model predictions do not match original model predictions"


def test_compiled_dlgn_model_is_faster_than_cpu(tmp_path):
    """Profile a dense model and verify compiled inference is faster than PyTorch CPU inference."""
    torch.manual_seed(0)

    model = Dlgn(
        thresholds=None,
        binarization="dummy",
        binarization_kwargs={},
        in_dim=256,
        n_layers=3,
        neurons_per_layer=512,
        group_sum_method="learnable_linear",
        group_sum_kwargs={"k": 8},
        device="cpu",
        connections_kwargs={"init_method": "random"},
        parametrization_kwargs={"weight_init": "random"},
        parametrization='raw'
    )
    model.train(False)

    X = torch.randint(0, 2, (4096, 1, 16, 16)).float()
    X_compiled = X.bool().numpy()

    compiled_model = CompiledLogicNet(
        model=model, input_shape=(1, 16, 16), num_bits=64, cpu_compiler="gcc", verbose=False
    )
    compiled_model.compile(save_lib_path=str(tmp_path / "compiled_dlgn_model.so"), verbose=False)

    with torch.no_grad():
        preds = model(X)
    preds_compiled = compiled_model(X_compiled)
    assert np.allclose(preds.numpy(), preds_compiled, atol=1e-5), "Compiled model predictions do not match original model predictions"

    def run_cpu():
        with torch.no_grad():
            model(X)

    def run_compiled():
        compiled_model(X_compiled)

    cpu_time = _median_runtime(run_cpu)
    compiled_time = _median_runtime(run_compiled)

    print(f"CPU model median runtime: {cpu_time:.6f}s")
    print(f"Compiled model median runtime: {compiled_time:.6f}s")

    assert compiled_time < cpu_time, (
        f"Expected compiled inference to be faster than CPU inference, "
        f"but compiled={compiled_time:.6f}s and cpu={cpu_time:.6f}s"
    )
