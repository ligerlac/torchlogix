# Quick Start Guide

## Basic Usage

### Creating Logic Layers

```python
import torch
from torchlogix.layers import LogicDense, LogicConv2d

# Dense logic layer
dense_layer = LogicDense(
    in_dim=784,      # Input dimension
    out_dim=128,     # Output dimension
    tree_depth=3     # Depth of logic tree
)

# Convolutional logic layer
conv_layer = LogicConv2d(
    in_dim=(28, 28),           # Input image size
    num_kernels=16,            # Number of output channels
    tree_depth=3,              # Logic tree depth
    receptive_field_size=5,    # Kernel size
    padding=2                  # Padding
)
```

### Building a Complete Model

```python
import torch.nn as nn
from torchlogix.layers import LogicConv2d, LogicDense, OrPooling, GroupSum

class LogicNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            LogicConv2d(
                in_dim=(28, 28),
                channels=1,
                num_kernels=16,
                tree_depth=3,
                receptive_field_size=5,
                padding=2
            ),
            OrPooling(kernel_size=2, stride=2, padding=0),

            LogicConv2d(
                in_dim=(14, 14),
                channels=16,
                num_kernels=32,
                tree_depth=3,
                receptive_field_size=3,
                padding=1
            ),
            OrPooling(kernel_size=2, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            LogicDense(in_dim=32*7*7, out_dim=256, tree_depth=4),
            LogicDense(in_dim=256, out_dim=128, tree_depth=3),
            GroupSum(num_classes, tau=1.0)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create and use the model
model = LogicNet(num_classes=10)
x = torch.randn(32, 1, 28, 28)  # Batch of MNIST-like images
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]
```

### Using Pre-built Models

```python
from torchlogix.models import CNN

# Pre-configured CNN for MNIST-like tasks
model = CNN(class_count=10, tau=1.0)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Model Compilation

Convert trained models to optimized implementations:

```python
from torchlogix import CompiledLogicNet

# After training your model
compiled_model = CompiledLogicNet(
    model=trained_model,
    input_shape=(1, 28, 28),
    device='cuda'
)

# Use compiled model for inference
with torch.no_grad():
    fast_output = compiled_model(test_input)
```

## Key Concepts

### Logic Tree Depth
- Controls the complexity of logic operations
- Higher depth = more complex logic expressions
- Typical values: 2-5

### Receptive Field Size
- Size of the convolutional kernel
- Determines local connectivity
- Must be ≤ input dimensions

### Connection Types
- `"random"`: Random connections (allows duplicates)
- `"random-unique"`: Unique random connections
- Affects learning capacity and speed

## Next Steps

- Read the [Logic Gates Guide](logic_gates.md) to understand the underlying operations
- Check out [Examples](examples.md) for complete training scripts
- Explore the [API Reference](../api/torchlogix.rst) for detailed documentation