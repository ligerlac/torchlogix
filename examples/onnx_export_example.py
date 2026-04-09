"""Example demonstrating ONNX export of TorchLogix models.

This example shows how to:
1. Train a TorchLogix model (or load a trained one)
2. Enable export mode for ONNX/TorchScript compatibility
3. Export the model to ONNX format
4. Verify the exported model works correctly
"""

import torch
import torch.nn as nn

from torchlogix.layers.dense import LogicDense


class ComplexLogicModel(nn.Module):
    """Example model with custom forward method (residual connections, parallel branches)."""

    def __init__(self):
        super().__init__()
        # Main branch
        self.layer1 = LogicDense(10, 8, parametrization="raw")
        self.layer2 = LogicDense(8, 6, parametrization="raw")

        # Parallel branch (out_dim * lut_rank >= in_dim, so 6 * 2 >= 10)
        self.parallel1 = LogicDense(10, 6, parametrization="raw")

        # Merge (6 + 6 = 12 inputs, so out_dim >= 6)
        self.merge = LogicDense(12, 6, parametrization="raw")

    def forward(self, x):
        # Main branch
        out1 = self.layer1(x)
        out2 = self.layer2(out1)

        # Parallel branch
        out_parallel = self.parallel1(x)

        # Concatenate and merge
        merged = torch.cat([out2, out_parallel], dim=1)
        return self.merge(merged)


def enable_export_mode_recursive(model):
    """Recursively enable export mode on all layers in a model.

    Args:
        model: PyTorch model containing LogicDense or other logic layers
    """
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)
            print(f"  Enabled export mode on {module.__class__.__name__}")


def main():
    print("=" * 80)
    print("TorchLogix ONNX Export Example")
    print("=" * 80)

    # 1. Create and "train" a model (here we just initialize it)
    print("\n1. Creating model...")
    model = ComplexLogicModel()
    print(f"   Model architecture:")
    print(f"   - Main branch: 10 -> 8 -> 6")
    print(f"   - Parallel branch: 10 -> 6")
    print(f"   - Merge: 12 (6+6) -> 6")

    # 2. Enable export mode
    print("\n2. Enabling export mode...")
    enable_export_mode_recursive(model)

    # 3. Set model to eval mode (required for discrete operations)
    model.eval()
    print("\n3. Model set to eval mode")

    # 4. Create dummy input for export
    batch_size = 1
    input_dim = 10
    dummy_input = torch.randint(0, 2, (batch_size, input_dim)).float()
    print(f"\n4. Created dummy input: shape={dummy_input.shape}, dtype={dummy_input.dtype}")

    # 5. Test forward pass before export
    print("\n5. Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output: {output}")

    # 6. Export to ONNX
    output_path = "logic_model.onnx"
    print(f"\n6. Exporting to ONNX: {output_path}")

    try:
        torch.onnx.export(
            model,                              # Model to export
            dummy_input,                        # Dummy input
            output_path,                        # Output file path
            export_params=True,                 # Export learned parameters
            opset_version=13,                   # ONNX opset version (13+ supports boolean ops)
            do_constant_folding=True,           # Optimize constant operations
            input_names=['input'],              # Input tensor name
            output_names=['output'],            # Output tensor name
            dynamic_axes={                      # Allow dynamic batch size
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"   ✓ Successfully exported to {output_path}")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return

    # 7. Verify exported model (requires onnx and onnxruntime)
    print("\n7. Verifying exported model...")
    try:
        import onnx
        import onnxruntime as ort

        # Load and check ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model is valid")

        # Run inference with ONNX Runtime
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare with PyTorch output
        print(f"   PyTorch output: {output}")
        print(f"   ONNX output:    {ort_outputs[0]}")

        if torch.allclose(output, torch.from_numpy(ort_outputs[0]), atol=1e-6):
            print("   ✓ Outputs match!")
        else:
            print("   ⚠ Outputs differ slightly (this may be expected)")

    except ImportError:
        print("   ⚠ onnx/onnxruntime not installed, skipping verification")
        print("   Install with: pip install onnx onnxruntime")
    except Exception as e:
        print(f"   ✗ Verification failed: {e}")

    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)
    print(f"\nExported model saved to: {output_path}")
    print("\nYou can now use this ONNX model with:")
    print("  - ONNX Runtime")
    print("  - TensorRT")
    print("  - CoreML")
    print("  - Other ONNX-compatible frameworks")


if __name__ == "__main__":
    main()
