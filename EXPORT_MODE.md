# Export Mode for ONNX/TorchScript

This document describes the **export mode** feature that enables TorchLogix models to be exported to ONNX and traced with TorchScript.

## Problem

The original numpy-wip branch attempted to make the forward pass compatible with 3rd party tracers by using only numpy-compatible operations. However, this resulted in a **~10% speed loss** in training scenarios. The tradeoff between tracer compatibility and training performance was unacceptable.

## Solution

Instead of sacrificing training performance, we implemented an **export mode** that:
- Keeps the fast, optimized PyTorch path for training (main branch performance)
- Enables tracer-friendly operations only when exporting
- Works with arbitrary model architectures (not just Sequential)
- Requires no code duplication

## How It Works

### Key Components

1. **`apply_luts_vectorized_export_mode()`** in `functional.py`
   - Tracer-friendly version of LUT application
   - Uses `torch.where()` instead of Python for loops
   - Unrolls all 16 boolean operations explicitly
   - Fully traceable by ONNX and TorchScript

2. **`export_mode` parameter** in parametrization classes
   - Added to `LUTParametrization` base class
   - Propagated to `RawLUTParametrization`, `WarpLUTParametrization`, `LightLUTParametrization`
   - When enabled in eval mode, uses `apply_luts_vectorized_export_mode()` instead of regular operations

3. **`set_export_mode()` method** in layers
   - Added to `LogicBase` base class
   - Inherited by all logic layers (`LogicDense`, `LogicConv2d`, etc.)
   - Enables/disables export mode on the layer's parametrization

## Usage

### Basic Example

```python
import torch
from torchlogix.layers.dense import LogicDense

# 1. Create and train model (normal workflow)
model = LogicDense(10, 5, parametrization="raw")
# ... train the model ...

# 2. Enable export mode
model.set_export_mode(True)
model.eval()

# 3. Export to ONNX
dummy_input = torch.randint(0, 2, (1, 10)).float()
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['output']
)
```

### Complex Models with Custom Forward

Export mode works with arbitrary architectures, including:
- Residual connections
- Parallel branches
- Custom forward methods

```python
import torch.nn as nn
from torchlogix.layers.dense import LogicDense

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = LogicDense(10, 6, parametrization="raw")
        self.branch2 = LogicDense(10, 6, parametrization="raw")
        self.merge = LogicDense(12, 6, parametrization="raw")

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        merged = torch.cat([out1, out2], dim=1)
        return self.merge(merged)

model = ComplexModel()

# Enable export mode recursively on all layers
for module in model.modules():
    if hasattr(module, 'set_export_mode'):
        module.set_export_mode(True)

model.eval()

# Export to ONNX
dummy_input = torch.randint(0, 2, (1, 10)).float()
torch.onnx.export(model, dummy_input, "complex_model.onnx")
```

## Benefits

| Feature | Export Mode | numpy-wip Branch | Main Branch |
|---------|-------------|------------------|-------------|
| **Training Speed** | ✅ Full speed | ❌ 10% slower | ✅ Full speed |
| **Arbitrary Architectures** | ✅ Yes (tracing) | ✅ Yes | ✅ Yes |
| **Code Duplication** | ✅ No duplication | ✅ No duplication | N/A |
| **ONNX Export** | ✅ Yes | ✅ Yes | ❌ No |
| **TorchScript** | ✅ Yes | ✅ Yes | ❌ No |
| **3rd Party Tools** | ✅ ONNX ecosystem | ✅ Numpy tracers | ❌ Limited |
| **Maintenance** | ✅ Single impl | ⚠️ Always slower | N/A |

## Implementation Details

### Boolean Operations

The export mode uses pure boolean operations that tracers can understand:

```python
# Example: AND operation (LUT ID = 1)
result = torch.where(
    lut_ids == 1,
    (a_bool & b_bool).float(),
    result
)
```

All 16 boolean operations are unrolled this way, making the computation graph explicit and traceable.

### Mode Selection

- **Training mode**: Always uses fast, optimized operations (regardless of export_mode flag)
- **Eval mode + export_mode=False**: Uses regular discrete operations (einsum-based)
- **Eval mode + export_mode=True**: Uses tracer-friendly operations (torch.where-based)

### Supported Parametrizations

Currently, export mode supports:
- `RawLUTParametrization` (lut_rank=2)
- `WarpLUTParametrization` (lut_rank=2)
- `LightLUTParametrization` (lut_rank=2)

Higher lut_rank support can be added if needed.

## Testing

Comprehensive tests are available in `tests/test_export_mode.py`:

- ✅ All 16 boolean operations
- ✅ Batch processing
- ✅ Mixed operations
- ✅ All parametrization types
- ✅ Layer-level export mode
- ✅ Complex model architectures
- ✅ TorchScript tracing
- ✅ ONNX export

Run tests with:
```bash
pytest tests/test_export_mode.py -v
```

## Example

A complete ONNX export example is available in `examples/onnx_export_example.py`:

```bash
python examples/onnx_export_example.py
```

This demonstrates:
- Creating a complex model with residual/parallel branches
- Enabling export mode
- Exporting to ONNX
- Verifying the exported model

## Deployment Workflow

```
┌─────────────────────────────────────────────────────┐
│ Training (main branch - full speed)                 │
│ ├─ Use regular forward pass                         │
│ ├─ Einsum-based operations                          │
│ └─ ~10% faster than numpy-wip                       │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Export Mode Activation                              │
│ ├─ model.eval()                                     │
│ ├─ model.set_export_mode(True)                      │
│ └─ Switches to tracer-friendly operations           │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ ONNX Export                                         │
│ ├─ torch.onnx.export(...)                           │
│ ├─ Pure boolean operations                          │
│ └─ Traceable computation graph                      │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Deployment                                          │
│ ├─ ONNX Runtime                                     │
│ ├─ TensorRT                                         │
│ ├─ CoreML                                           │
│ └─ Other ONNX-compatible frameworks                 │
└─────────────────────────────────────────────────────┘
```

## Comparison to Other Approaches

### vs. numpy-wip Branch
- **Advantage**: No training speed sacrifice (10% faster)
- **Advantage**: Uses optimized PyTorch operations for training
- **Same**: Works with arbitrary architectures
- **Same**: Single forward implementation (no duplication)

### vs. compiled_model.py (C/Verilog export)
- **Advantage**: Works with arbitrary architectures (not just Sequential)
- **Advantage**: No code duplication (doesn't re-implement forward pass)
- **Advantage**: Standard ONNX format (wider ecosystem support)
- **Different**: ONNX vs. C/Verilog (complementary deployment targets)

## Future Enhancements

Potential improvements:
- [ ] Support for higher lut_rank (4, 6) in export mode
- [ ] Quantization support in ONNX export
- [ ] Automatic opset version detection
- [ ] Export to other formats (TorchScript SavedModel, CoreML)

## References

- ONNX Documentation: https://onnx.ai/
- PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
- TorchScript: https://pytorch.org/docs/stable/jit.html
