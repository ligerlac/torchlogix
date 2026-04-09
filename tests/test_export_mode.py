"""Tests for export mode functionality (ONNX/TorchScript tracing)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchlogix.layers import LogicDense, LogicConv2d, OrPooling2d, GroupSum
from torchlogix.functional import apply_luts_vectorized_export_mode


class TestApplyLutsVectorizedExportMode:
    """Test the tracer-friendly LUT application function."""

    def test_basic_operations(self):
        """Test that all 16 logic operations work correctly."""
        # Create simple inputs covering all truth table cases
        # Each column represents one truth table entry: (0,0), (0,1), (1,0), (1,1)
        a = torch.tensor([[False, False, True, True]])
        b = torch.tensor([[False, True, False, True]])

        # Expected outputs are the truth table for each operation
        # Applied to inputs (a,b) = [(0,0), (0,1), (1,0), (1,1)]
        expected_outputs = {
            0: [False, False, False, False],  # FALSE
            1: [False, False, False, True],  # AND
            2: [False, False, True, False],  # A AND NOT B
            3: [False, False, True, True],  # A
            4: [False, True, False, False],  # B AND NOT A
            5: [False, True, False, True],  # B
            6: [False, True, True, False],  # XOR
            7: [False, True, True, True],  # OR
            8: [True, False, False, False],  # NOR
            9: [True, False, False, True],  # XNOR
            10: [True, False, True, False],  # NOT B
            11: [True, False, True, True],  # B IMPLIES A
            12: [True, True, False, False],  # NOT A
            13: [True, True, False, True],  # A IMPLIES B
            14: [True, True, True, False],  # NAND
            15: [True, True, True, True],  # TRUE
        }

        for lut_id in range(16):
            lut_ids = torch.tensor([lut_id, lut_id, lut_id, lut_id])
            result = apply_luts_vectorized_export_mode(a, b, lut_ids)
            expected = torch.tensor([expected_outputs[lut_id]])
            assert torch.allclose(result, expected), \
                f"LUT {lut_id} failed: expected {expected}, got {result}"

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        batch_size = 8
        num_neurons = 4

        # Create random inputs
        a = torch.rand(batch_size, num_neurons) > 0.5
        b = torch.rand(batch_size, num_neurons) > 0.5

        # Use AND operation (LUT ID = 1)
        lut_ids = torch.ones(num_neurons, dtype=torch.long)
        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        # AND operation: result should be 1 only where both inputs are 1
        expected = a & b
        assert torch.allclose(result, expected)

    def test_mixed_operations(self):
        """Test multiple different operations in parallel."""
        # Input pattern: (a,b) for each neuron
        # neuron 0: (0,0), neuron 1: (1,0), neuron 2: (0,1), neuron 3: (1,1)
        a = torch.tensor([[False, True, False, True]])
        b = torch.tensor([[False, False, True, True]])

        # Use different operations for each neuron: AND, OR, XOR, NAND
        lut_ids = torch.tensor([1, 7, 6, 14])
        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        # Expected outputs:
        # neuron 0: AND(0,0) = 0
        # neuron 1: OR(1,0) = 1
        # neuron 2: XOR(0,1) = 1
        # neuron 3: NAND(1,1) = 0
        expected = torch.tensor([[False, True, True, False]])
        assert torch.allclose(result, expected)

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        batch_size = 4
        num_neurons = 3

        a = torch.rand(batch_size, num_neurons) > 0.5
        b = torch.rand(batch_size, num_neurons) > 0.5
        lut_ids = torch.tensor([1, 7, 6])  # AND, OR, XOR

        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        assert result.shape == (batch_size, num_neurons)

        # Verify each operation
        assert torch.allclose(result[:, 0], (a[:, 0] & b[:, 0]))  # AND
        assert torch.allclose(result[:, 1], (a[:, 1] | b[:, 1]))  # OR
        assert torch.allclose(result[:, 2], (a[:, 2] ^ b[:, 2]))  # XOR


class TestApplyLutsVectorizedExportModeNumpy:
    """Test the tracer-friendly LUT application function with NumPy backend."""

    def test_basic_operations_numpy(self):
        """Test that all 16 logic operations work correctly with NumPy arrays."""
        # Create simple inputs covering all truth table cases
        a = np.array([[False, False, True, True]])
        b = np.array([[False, True, False, True]])

        # Expected outputs are the truth table for each operation
        expected_outputs = {
            0: [False, False, False, False],  # FALSE
            1: [False, False, False, True],  # AND
            2: [False, False, True, False],  # A AND NOT B
            3: [False, False, True, True],  # A
            4: [False, True, False, False],  # B AND NOT A
            5: [False, True, False, True],  # B
            6: [False, True, True, False],  # XOR
            7: [False, True, True, True],  # OR
            8: [True, False, False, False],  # NOR
            9: [True, False, False, True],  # XNOR
            10: [True, False, True, False],  # NOT B
            11: [True, False, True, True],  # B IMPLIES A
            12: [True, True, False, False],  # NOT A
            13: [True, True, False, True],  # A IMPLIES B
            14: [True, True, True, False],  # NAND
            15: [True, True, True, True],  # TRUE
        }

        for lut_id in range(16):
            lut_ids = np.array([lut_id, lut_id, lut_id, lut_id])
            result = apply_luts_vectorized_export_mode(a, b, lut_ids)
            expected = np.array([expected_outputs[lut_id]])
            assert np.array_equal(result, expected), \
                f"LUT {lut_id} failed: expected {expected}, got {result}"
            # Verify output is numpy array
            assert isinstance(result, np.ndarray)

    def test_batch_processing_numpy(self):
        """Test that batch processing works correctly with NumPy."""
        batch_size = 8
        num_neurons = 4

        # Create random inputs
        a = np.random.rand(batch_size, num_neurons) > 0.5
        b = np.random.rand(batch_size, num_neurons) > 0.5

        # Use AND operation (LUT ID = 1)
        lut_ids = np.ones(num_neurons, dtype=int)
        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        # AND operation: result should be 1 only where both inputs are 1
        expected = (a & b)
        assert np.array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_mixed_operations_numpy(self):
        """Test multiple different operations in parallel with NumPy."""
        a = np.array([[False, True, False, True]])
        b = np.array([[False, False, True, True]])

        # Use different operations for each neuron: AND, OR, XOR, NAND
        lut_ids = np.array([1, 7, 6, 14])
        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        expected = np.array([[False, True, True, False]])
        assert np.array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_broadcasting_numpy(self):
        """Test that broadcasting works correctly with NumPy."""
        batch_size = 4
        num_neurons = 3

        a = np.random.rand(batch_size, num_neurons) > 0.5
        b = np.random.rand(batch_size, num_neurons) > 0.5
        lut_ids = np.array([1, 7, 6])  # AND, OR, XOR

        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        assert result.shape == (batch_size, num_neurons)
        assert isinstance(result, np.ndarray)

        # Verify each operation
        assert np.array_equal(result[:, 0], (a[:, 0] & b[:, 0]))  # AND
        assert np.array_equal(result[:, 1], (a[:, 1] | b[:, 1]))  # OR
        assert np.array_equal(result[:, 2], (a[:, 2] ^ b[:, 2]))  # XOR


class TestTorchNumpyEquivalence:
    """Test that torch and numpy backends produce identical results."""

    def test_all_operations_equivalence(self):
        """Test that all 16 operations produce identical results in both backends."""
        # Create test inputs
        a_torch = torch.tensor([[False, False, True, True]])
        b_torch = torch.tensor([[False, True, False, True]])
        a_numpy = np.array([[False, False, True, True]])
        b_numpy = np.array([[False, True, False, True]])

        for lut_id in range(16):
            lut_ids_torch = torch.tensor([lut_id, lut_id, lut_id, lut_id])
            lut_ids_numpy = np.array([lut_id, lut_id, lut_id, lut_id])

            result_torch = apply_luts_vectorized_export_mode(a_torch, b_torch, lut_ids_torch)
            result_numpy = apply_luts_vectorized_export_mode(a_numpy, b_numpy, lut_ids_numpy)

            # Convert torch result to numpy for comparison
            assert np.array_equal(result_torch.numpy(), result_numpy), \
                f"LUT {lut_id} results differ between torch and numpy"

    def test_batch_equivalence(self):
        """Test torch/numpy equivalence on batch processing."""
        batch_size = 8
        num_neurons = 4

        # Create identical inputs in both frameworks
        np.random.seed(42)
        a_np = np.random.rand(batch_size, num_neurons) > 0.5
        b_np = np.random.rand(batch_size, num_neurons) > 0.5
        lut_ids_np = np.array([1, 7, 6, 14])  # AND, OR, XOR, NAND

        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        lut_ids_torch = torch.from_numpy(lut_ids_np)

        result_torch = apply_luts_vectorized_export_mode(a_torch, b_torch, lut_ids_torch)
        result_numpy = apply_luts_vectorized_export_mode(a_np, b_np, lut_ids_np)

        assert np.array_equal(result_torch.numpy(), result_numpy)

    @pytest.mark.parametrize("param_type", ["raw", "warp", "light"])
    def test_numpy_torch_equivalence_dense(self, param_type):
        batch_size = 32
        in_dim = 8
        out_dim = 16

        layer = LogicDense(in_dim, out_dim, parametrization=param_type)
        layer.set_export_mode()

        x_np = np.random.randint(0, 2, (batch_size, in_dim), dtype=bool)
        x_torch = torch.from_numpy(x_np)

        result_torch = layer(x_torch)
        result_numpy = layer(x_np)

        # Compare results
        assert np.allclose(result_torch.numpy(), result_numpy, atol=1e-6)

    @pytest.mark.parametrize("param_type", ["raw", "warp", "light"])
    def test_numpy_torch_equivalence_conv(self, param_type):
        batch_size = 16
        in_dim = 16
        channels = 3
        num_kernels = 8

        layer = LogicConv2d(in_dim=in_dim, channels=channels, num_kernels=num_kernels, receptive_field_size=3, tree_depth=2)
        layer.set_export_mode()

        x_np = np.random.randint(0, 2, (batch_size, channels, in_dim, in_dim), dtype=bool)
        x_torch = torch.from_numpy(x_np)

        result_torch = layer(x_torch)
        result_numpy = layer(x_np)

        assert np.allclose(result_torch.numpy(), result_numpy, atol=1e-6)


class TestLayerExportMode:
    """Test export mode in logic layers."""

    def test_logic_dense_export_mode(self):
        """Test LogicDense layer with export mode."""
        in_dim = 10
        out_dim = 5

        # Create layer
        layer = LogicDense(
            in_dim=in_dim,
            out_dim=out_dim,
            parametrization="raw"
        )

        # Enable export mode
        layer.set_export_mode()
        assert layer.export_mode is True

        # Disable export mode
        layer.set_export_mode(False)
        assert layer.export_mode is False


    @pytest.mark.parametrize("param_type", ["raw", "warp", "light"])
    def test_export_mode_equivalence_in_dense_layer(self, param_type):
        """Test that layer produces same results with and without export mode on binary inputs."""
        in_dim = 32
        out_dim = 64
        batch_size = 128

        # Create layer
        layer = LogicDense(
            in_dim=in_dim,
            out_dim=out_dim,
            parametrization=param_type
        )
        layer.eval()

        # Create BINARY input (this is what logic networks expect at inference)
        x = torch.randint(0, 2, (batch_size, in_dim)).float()

        # Forward pass without export mode
        layer.set_export_mode(False)
        result_regular = layer(x)

        # Forward pass with export mode
        layer.set_export_mode(True)
        result_export = layer(x.bool())

        # Results should be identical for binary inputs
        assert torch.allclose(result_regular, result_export.float(), atol=1e-6)


    @pytest.mark.parametrize("param_type", ["raw", "warp", "light"])
    def test_export_mode_equivalence_in_conv_layer(self, param_type):
        """Test that layer produces same results with and without export mode on binary inputs."""
        in_dim = 32
        channels = 3
        num_kernels = 16
        batch_size = 128

        # Create layer
        layer = LogicConv2d(in_dim=in_dim, channels=channels, num_kernels=num_kernels, receptive_field_size=3, tree_depth=3)
        layer.eval()

        # Create BINARY input (this is what logic networks expect at inference)
        x = torch.randint(0, 2, (batch_size, channels, in_dim, in_dim)).float()

        # Forward pass without export mode
        layer.set_export_mode(False)
        result_regular = layer(x)

        # Forward pass with export mode
        layer.set_export_mode(True)
        result_export = layer(x.bool())

        # Results should be identical for binary inputs
        assert torch.allclose(result_regular, result_export.float(), atol=1e-6)


class TestComplexModelExportMode:
    """Test export mode with complex model architectures."""

    def test_sequential_model(self):
        """Test export mode with sequential model."""
        model = nn.Sequential(
            LogicDense(10, 8, parametrization="raw"),
            LogicDense(8, 4, parametrization="raw"),
            LogicDense(4, 2, parametrization="raw")
        )

        x = torch.rand(5, 10).bool()

        model.eval()
        result_regular = model(x)

        # Enable export mode on all layers
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        result_export = model(x)

        assert torch.allclose(result_regular, result_export.float(), atol=1e-6)


    def test_custom_forward_model(self):
        """Test export mode with custom forward method (residual connections)."""
        class ResidualLogicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = LogicDense(10, 10, parametrization="raw")
                self.layer2 = LogicDense(10, 10, parametrization="raw")
                self.layer3 = LogicDense(10, 5, parametrization="raw")

            def forward(self, x):
                # Residual connection
                out1 = self.layer1(x)
                out2 = self.layer2(out1)
                residual = out1 * out2
                return self.layer3(residual)
            
        x = torch.rand(3, 10).bool()
            
        model = ResidualLogicModel()
        model.eval()
        result_regular = model(x)

        # Enable export mode
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        result_export = model(x)

        assert torch.allclose(result_regular, result_export.float(), atol=1e-6)


    def test_parallel_branches_model(self):
        """Test export mode with parallel branches."""
        class ParallelLogicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch1 = LogicDense(10, 5, parametrization="raw")
                self.branch2 = LogicDense(10, 5, parametrization="raw")
                # Merged input is 10 (5+5), output is 6 (satisfies out_dim*lut_rank >= in_dim)
                self.merge = LogicDense(10, 6, parametrization="raw")

            def forward(self, x):
                # Parallel branches
                out1 = self.branch1(x)
                out2 = self.branch2(x)
                # Concatenate outputs
                merged = torch.cat([out1, out2], dim=1)
                return self.merge(merged)

        x = torch.rand(4, 10).bool()

        model = ParallelLogicModel()

        model.eval()
        result_regular = model(x)

        # Enable export mode
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        result_export = model(x)

        assert torch.allclose(result_regular, result_export.float(), atol=1e-6)


class TestTorchScriptTracing:
    """Test that export mode enables TorchScript tracing."""

    def test_simple_layer_tracing(self):
        """Test that a simple layer can be traced with TorchScript."""
        layer = LogicDense(8, 4, parametrization="raw")
        layer.set_export_mode(True)
        layer.eval()

        x = torch.rand(2, 8).bool()

        traced = torch.jit.trace(layer, x)

        # Test that traced model works
        result_original = layer(x)
        result_traced = traced(x)

        print("Original result:", result_original)
        print("Traced result:", result_traced)

        assert torch.allclose(result_original, result_traced, atol=1e-6)


    def test_complex_model_tracing(self):
        """Test that a complex model can be traced with TorchScript."""
        model = nn.Sequential(
            LogicConv2d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2),
            OrPooling2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),  # 3*3*8 = 72
            LogicDense(72, 64, parametrization="raw"),
            LogicDense(64, 50, parametrization="raw"),
            GroupSum(10),
        )

        # Enable export mode
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        model.eval()

        x = torch.rand(128, 3, 8, 8).bool()

        # Trace the model
        traced = torch.jit.trace(model, x)

        # Test that traced model works
        result_original = model(x)

        result_traced = traced(x)

        assert torch.allclose(result_original, result_traced, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
