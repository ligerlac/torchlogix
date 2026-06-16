# Quick Start

For a complete end-to-end walkthrough — training, evaluating, compiling, and
exporting to C/Verilog — see the
[MNIST example notebook](../../examples/mnist_example.ipynb).

This page shows the minimal API surface for building and exporting a model.

---

## Define a model

```python
import torch.nn as nn
from torchlogix.layers import LogicConv2d, OrPooling2d, LogicDense, GroupSum

model = nn.Sequential(
    LogicConv2d(in_dim=28, channels=1, num_kernels=16, receptive_field_size=3),
    OrPooling2d(kernel_size=2, stride=2),
    nn.Flatten(),
    LogicDense(16 * 13 * 13, 4_000),
    LogicDense(4_000, 4_000),
    GroupSum(k=10),
)
```

## Train

Logic layers are standard `nn.Module` objects — use any PyTorch training loop:

```python
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for x, y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

## Convert to Circuit and compile

> `Circuit` requires **binary inputs** — binarize at the dataset level before
> passing data to the model. Binarization layers are not exported.

```python
from torchlogix import Circuit
from torchlogix.utils import set_export_mode

set_export_mode(model)                                   # required before tracing
circuit = Circuit.from_model(model, input_shape=(1, 28, 28))
circuit.simplify()                                       # prune dead gates
circuit.compile()                                        # JIT to a C shared library

# Fast inference on boolean numpy arrays
import numpy as np
x_np = x.numpy().astype(np.bool_)
scores = circuit(x_np, use_compiled=True)
```

## Export to C or Verilog

```python
circuit.write_c_code("circuit.c")          # self-contained C99, no dependencies
circuit.write_verilog_code("circuit.v")    # combinational RTL module
```
