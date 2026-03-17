"""Tests for Verilog generation functionality."""

import pytest
import torch
import torch.nn as nn
from torchlogix.layers import LogicDense, LogicConv2d, GroupSum
from torchlogix import CompiledLogicNet
from torchlogix.hdl_generator import gate_id_to_verilog, GATE_TRUTH_TABLES


class TestGateMappings:
    """Test gate ID to Verilog mapping."""

    def test_gate_id_to_verilog_all_gates(self):
        """Test that all 16 gates map correctly."""
        for gate_id in range(16):
            result = gate_id_to_verilog(gate_id, "a", "b")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_gate_constants(self):
        """Test constant gates (0 and 15)."""
        assert gate_id_to_verilog(0, "x", "y") == "1'b0"
        assert gate_id_to_verilog(15, "x", "y") == "1'b1"

    def test_gate_and(self):
        """Test AND gate."""
        result = gate_id_to_verilog(1, "inp[0]", "inp[1]")
        assert "inp[0]" in result
        assert "inp[1]" in result
        assert "&" in result

    def test_gate_or(self):
        """Test OR gate."""
        result = gate_id_to_verilog(7, "inp[0]", "inp[1]")
        assert "inp[0]" in result
        assert "inp[1]" in result
        assert "|" in result

    def test_gate_xor(self):
        """Test XOR gate."""
        result = gate_id_to_verilog(6, "inp[0]", "inp[1]")
        assert "inp[0]" in result
        assert "inp[1]" in result
        assert "^" in result

    def test_gate_passthrough_a(self):
        """Test passthrough A."""
        result = gate_id_to_verilog(3, "inp[0]", "inp[1]")
        assert result == "inp[0]"

    def test_gate_passthrough_b(self):
        """Test passthrough B."""
        result = gate_id_to_verilog(5, "inp[0]", "inp[1]")
        assert result == "inp[1]"


class TestVerilogGeneration:
    """Test complete Verilog generation."""

    def test_simple_linear_model(self):
        """Test Verilog generation for simple LogicDense model."""
        model = nn.Sequential(
            LogicDense(4, 2, connections="fixed", device="cpu"),
            GroupSum(2)
        )

        compiled = CompiledLogicNet(model, input_shape=(4,), use_bitpacking=False, num_bits=1)
        verilog = compiled.get_verilog_code(module_name="test_module")

        # Check basic structure
        assert "module test_module" in verilog
        assert "input wire" in verilog
        assert "output wire" in verilog
        assert "endmodule" in verilog

        # Check that it contains assign statements
        assert "assign" in verilog

    def test_multi_layer_model(self):
        """Test Verilog generation for multi-layer model."""
        model = nn.Sequential(
            LogicDense(8, 4, connections="fixed", device="cpu"),
            LogicDense(4, 2, connections="fixed", device="cpu"),
            GroupSum(2)
        )

        compiled = CompiledLogicNet(model, input_shape=(8,), use_bitpacking=False, num_bits=1)
        verilog = compiled.get_verilog_code(module_name="multi_layer")

        # Check for multiple layers
        assert "Layer 0" in verilog
        assert "Layer 1" in verilog

        # Check for intermediate wires
        assert "layer_0_out" in verilog

    def test_conv_model(self):
        """Test Verilog generation for model with LogicConv2d."""
        model = nn.Sequential(
            LogicConv2d(in_dim=8, channels=2, num_kernels=2, receptive_field_size=3, tree_depth=2),
            GroupSum(2)
        )

        compiled = CompiledLogicNet(model, input_shape=(2, 8, 8), use_bitpacking=False, num_bits=1)
        verilog = compiled.get_verilog_code(module_name="conv_model")

        # Check for convolutional layer
        assert "Layer 0" in verilog
        assert "conv" in verilog


    def test_export_hdl_creates_file(self, tmp_path):
        """Test that export_hdl creates a file."""
        model = nn.Sequential(
            LogicDense(4, 2, connections="fixed", device="cpu"),
            GroupSum(2)
        )

        compiled = CompiledLogicNet(model, input_shape=(4,), use_bitpacking=False, num_bits=1)
        output_dir = str(tmp_path / "verilog_output")

        compiled.export_hdl(output_dir, module_name="exported_net", format="verilog")

        # Check file exists
        output_file = tmp_path / "verilog_output" / "exported_net.v"
        assert output_file.exists()

        # Check content
        content = output_file.read_text()
        assert "module exported_net" in content
        assert "endmodule" in content


class TestCompiledLogicNet:
    """Test CompiledLogicNet Verilog methods."""

    def test_get_gate_verilog_method(self):
        """Test get_gate_verilog method on CompiledLogicNet."""
        model = nn.Sequential(LogicDense(4, 2, device="cpu"), GroupSum(2))
        compiled = CompiledLogicNet(model, input_shape=(4,), use_bitpacking=False, num_bits=1)

        # Test AND gate
        result = compiled.get_gate_verilog("a", "b", 1)
        assert "&" in result

        # Test OR gate
        result = compiled.get_gate_verilog("x", "y", 7)
        assert "|" in result


class TestTruthTableValidation:
    """Validate that gate mappings match truth tables."""

    def test_truth_tables_exist(self):
        """Test that truth tables are defined for all gates."""
        for gate_id in range(16):
            assert gate_id in GATE_TRUTH_TABLES
            truth_table = GATE_TRUTH_TABLES[gate_id]
            assert len(truth_table) == 4  # [AB=00, AB=01, AB=10, AB=11]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
