"""Tests for export mode functionality (ONNX/TorchScript tracing)."""

import pytest
import torch
import torch.nn as nn

from torchlogix.layers.dense import LogicDense
from torchlogix.functional import apply_luts_vectorized_export_mode
from torchlogix.parametrization import (
    RawLUTParametrization,
    WarpLUTParametrization,
    LightLUTParametrization
)


class TestApplyLutsVectorizedExportMode:
    """Test the tracer-friendly LUT application function."""

    def test_basic_operations(self):
        """Test that all 16 logic operations work correctly."""
        # Create simple inputs covering all truth table cases
        # Each column represents one truth table entry: (0,0), (0,1), (1,0), (1,1)
        a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        b = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

        # Expected outputs are the truth table for each operation
        # Applied to inputs (a,b) = [(0,0), (0,1), (1,0), (1,1)]
        expected_outputs = {
            0: [0.0, 0.0, 0.0, 0.0],  # FALSE
            1: [0.0, 0.0, 0.0, 1.0],  # AND
            2: [0.0, 0.0, 1.0, 0.0],  # A AND NOT B
            3: [0.0, 0.0, 1.0, 1.0],  # A
            4: [0.0, 1.0, 0.0, 0.0],  # B AND NOT A
            5: [0.0, 1.0, 0.0, 1.0],  # B
            6: [0.0, 1.0, 1.0, 0.0],  # XOR
            7: [0.0, 1.0, 1.0, 1.0],  # OR
            8: [1.0, 0.0, 0.0, 0.0],  # NOR
            9: [1.0, 0.0, 0.0, 1.0],  # XNOR
            10: [1.0, 0.0, 1.0, 0.0],  # NOT B
            11: [1.0, 0.0, 1.0, 1.0],  # B IMPLIES A
            12: [1.0, 1.0, 0.0, 0.0],  # NOT A
            13: [1.0, 1.0, 0.0, 1.0],  # A IMPLIES B
            14: [1.0, 1.0, 1.0, 0.0],  # NAND
            15: [1.0, 1.0, 1.0, 1.0],  # TRUE
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
        result = apply_luts_vectorized_export_mode(a.float(), b.float(), lut_ids)

        # AND operation: result should be 1 only where both inputs are 1
        expected = (a & b).float()
        assert torch.allclose(result, expected)

    def test_mixed_operations(self):
        """Test multiple different operations in parallel."""
        # Input pattern: (a,b) for each neuron
        # neuron 0: (0,0), neuron 1: (1,0), neuron 2: (0,1), neuron 3: (1,1)
        a = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        # Use different operations for each neuron: AND, OR, XOR, NAND
        lut_ids = torch.tensor([1, 7, 6, 14])
        result = apply_luts_vectorized_export_mode(a, b, lut_ids)

        # Expected outputs:
        # neuron 0: AND(0,0) = 0
        # neuron 1: OR(1,0) = 1
        # neuron 2: XOR(0,1) = 1
        # neuron 3: NAND(1,1) = 0
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        batch_size = 4
        num_neurons = 3

        a = torch.rand(batch_size, num_neurons) > 0.5
        b = torch.rand(batch_size, num_neurons) > 0.5
        lut_ids = torch.tensor([1, 7, 6])  # AND, OR, XOR

        result = apply_luts_vectorized_export_mode(a.float(), b.float(), lut_ids)

        assert result.shape == (batch_size, num_neurons)

        # Verify each operation
        assert torch.allclose(result[:, 0], (a[:, 0] & b[:, 0]).float())  # AND
        assert torch.allclose(result[:, 1], (a[:, 1] | b[:, 1]).float())  # OR
        assert torch.allclose(result[:, 2], (a[:, 2] ^ b[:, 2]).float())  # XOR


class TestParametrizationExportMode:
    """Test export mode in parametrization classes."""

    @pytest.mark.parametrize("param_class", [
        RawLUTParametrization,
        WarpLUTParametrization,
        LightLUTParametrization
    ])
    def test_export_mode_flag(self, param_class):
        """Test that export_mode flag is set correctly."""
        # Create parametrization without export mode
        param = param_class(lut_rank=2, export_mode=False)
        assert param.export_mode is False

        # Create parametrization with export mode
        param_export = param_class(lut_rank=2, export_mode=True)
        assert param_export.export_mode is True

        # Test set_export_mode method
        param.set_export_mode(True)
        assert param.export_mode is True
        param.set_export_mode(False)
        assert param.export_mode is False

    @pytest.mark.parametrize("param_class", [
        RawLUTParametrization,
        WarpLUTParametrization,
        LightLUTParametrization
    ])
    def test_export_mode_forward(self, param_class):
        """Test that export mode uses tracer-friendly operations."""
        num_neurons = 4
        batch_size = 2

        # Create parametrization with export mode
        param = param_class(lut_rank=2, export_mode=True)
        weights = param.init_weights(num_neurons, "cpu")

        # Create inputs
        x = torch.rand(batch_size, 2, num_neurons)

        # Forward pass in eval mode should use export-friendly operations
        result = param.forward(x, weights, training=False, contraction='n,bn->bn')

        assert result.shape == (batch_size, num_neurons)
        assert result.dtype == torch.float32

    def test_export_vs_regular_mode_equivalence(self):
        """Test that export mode produces same results as regular mode on binary inputs.

        Note: Export mode uses pure boolean operations (0/1 outputs), while regular mode
        uses weighted sums which can produce continuous values. They are only equivalent
        when inputs are binary and we compare the selected operation.
        """
        num_neurons = 8
        batch_size = 4

        # Create two parametrizations: one with export mode, one without
        param_regular = RawLUTParametrization(lut_rank=2, export_mode=False)
        param_export = RawLUTParametrization(lut_rank=2, export_mode=True)

        # Use same weights
        weights = param_regular.init_weights(num_neurons, "cpu")

        # Create BINARY inputs (0 or 1) - this is where they should match
        x = torch.randint(0, 2, (batch_size, 2, num_neurons)).float()

        # Forward pass in eval mode
        result_regular = param_regular.forward(x, weights, training=False, contraction='n,bn->bn')
        result_export = param_export.forward(x, weights, training=False, contraction='n,bn->bn')

        # Results should be identical for binary inputs
        assert torch.allclose(result_regular, result_export, atol=1e-6)


class TestLayerExportMode:
    """Test export mode in logic layers."""

    def test_logic_dense_export_mode(self):
        """Test LogicDense layer with export mode."""
        in_dim = 10
        out_dim = 5
        batch_size = 3

        # Create layer
        layer = LogicDense(
            in_dim=in_dim,
            out_dim=out_dim,
            parametrization="raw"
        )

        # Enable export mode
        layer.set_export_mode(True)
        assert layer.parametrization.export_mode is True

        # Test forward pass in eval mode
        layer.eval()
        x = torch.rand(batch_size, in_dim)
        result = layer(x)

        assert result.shape == (batch_size, out_dim)

        # Disable export mode
        layer.set_export_mode(False)
        assert layer.parametrization.export_mode is False

    def test_export_mode_equivalence_in_layer(self):
        """Test that layer produces same results with and without export mode on binary inputs."""
        in_dim = 8
        out_dim = 4
        batch_size = 2

        # Create layer
        layer = LogicDense(
            in_dim=in_dim,
            out_dim=out_dim,
            parametrization="raw"
        )
        layer.eval()

        # Create BINARY input (this is what logic networks expect at inference)
        x = torch.randint(0, 2, (batch_size, in_dim)).float()

        # Forward pass without export mode
        layer.set_export_mode(False)
        result_regular = layer(x)

        # Forward pass with export mode
        layer.set_export_mode(True)
        result_export = layer(x)

        # Results should be identical for binary inputs
        assert torch.allclose(result_regular, result_export, atol=1e-6)

    @pytest.mark.parametrize("param_type", ["raw", "warp", "light"])
    def test_all_parametrizations(self, param_type):
        """Test export mode works with all parametrization types."""
        in_dim = 6
        out_dim = 3

        layer = LogicDense(
            in_dim=in_dim,
            out_dim=out_dim,
            parametrization=param_type
        )

        layer.set_export_mode(True)
        layer.eval()

        x = torch.rand(2, in_dim)
        result = layer(x)

        assert result.shape == (2, out_dim)
        assert not torch.isnan(result).any()


class TestComplexModelExportMode:
    """Test export mode with complex model architectures."""

    def test_sequential_model(self):
        """Test export mode with sequential model."""
        model = nn.Sequential(
            LogicDense(10, 8, parametrization="raw"),
            LogicDense(8, 4, parametrization="raw"),
            LogicDense(4, 2, parametrization="raw")
        )

        # Enable export mode on all layers
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        model.eval()

        x = torch.rand(5, 10)
        result = model(x)

        assert result.shape == (5, 2)
        assert not torch.isnan(result).any()

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
                # Simple addition as "residual" (works with boolean logic)
                residual = out1 + out2
                return self.layer3(residual)

        model = ResidualLogicModel()

        # Enable export mode
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        model.eval()

        x = torch.rand(3, 10)
        result = model(x)

        assert result.shape == (3, 5)
        assert not torch.isnan(result).any()

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

        model = ParallelLogicModel()

        # Enable export mode
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        model.eval()

        x = torch.rand(4, 10)
        result = model(x)

        assert result.shape == (4, 6)
        assert not torch.isnan(result).any()


class TestTorchScriptTracing:
    """Test that export mode enables TorchScript tracing."""

    def test_simple_layer_tracing(self):
        """Test that a simple layer can be traced with TorchScript."""
        layer = LogicDense(8, 4, parametrization="raw")
        layer.set_export_mode(True)
        layer.eval()

        x = torch.rand(2, 8)

        # Trace the layer
        try:
            traced = torch.jit.trace(layer, x)

            # Test that traced model works
            result_original = layer(x)
            result_traced = traced(x)

            assert torch.allclose(result_original, result_traced, atol=1e-6)
        except Exception as e:
            pytest.fail(f"TorchScript tracing failed: {e}")

    def test_complex_model_tracing(self):
        """Test that a complex model can be traced with TorchScript."""
        model = nn.Sequential(
            LogicDense(10, 8, parametrization="raw"),
            LogicDense(8, 4, parametrization="raw")
        )

        # Enable export mode
        for module in model.modules():
            if hasattr(module, 'set_export_mode'):
                module.set_export_mode(True)

        model.eval()

        x = torch.rand(3, 10)

        # Trace the model
        try:
            traced = torch.jit.trace(model, x)

            # Test that traced model works
            result_original = model(x)
            result_traced = traced(x)

            assert torch.allclose(result_original, result_traced, atol=1e-6)
        except Exception as e:
            pytest.fail(f"TorchScript tracing failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
