# Installation Guide

## Requirements

- Python 3.6 or higher
- PyTorch 1.9.0 or higher

## Basic Installation

### Using pip

```bash
pip install torchlogix
```

### From Source

```bash
git clone https://github.com/ligerlac/torchlogix.git
cd torchlogix
pip install -e .
```

### Development Installation

For development with all dependencies:

```bash
git clone https://github.com/ligerlac/torchlogix.git
cd torchlogix
pip install -e .[dev]
```

## Conda Environment

For a complete environment setup:

```bash
conda env create -f environment.yml
conda activate torchlogix
pip install -e .
```

## Verification

Test your installation:

```python
import torch
import torchlogix

# Create a simple logic layer
layer = torchlogix.layers.LogicDense(in_dim=10, out_dim=5, tree_depth=2)
x = torch.randn(32, 10)
output = layer(x)
print(f"Output shape: {output.shape}")
```

## Supported Versions

- **Python**: 3.6+
- **PyTorch**: 1.9.0 - 1.13.x (tested)

For experiments, install additional dependencies:
```bash
pip install -r experiments/requirements.txt
```
