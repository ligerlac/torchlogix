# Installation Guide

## Requirements

- Python 3.6 or higher
- PyTorch 1.9.0 or higher
- CUDA toolkit (optional, for GPU acceleration)

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

## CUDA Extension Support

> ⚠️ **Important**: By default, `torchlogix` requires CUDA, the CUDA Toolkit (for compilation), and `torch>=1.9.0` (matching the CUDA version). CUDA can be disabled by setting a flag like so `export TORCHLOGIX_BUILD_CUDA_EXT=false` before running `pip install .`. Only the much slower pure Python implementation is available in that case.

**It is very important that the installed version of PyTorch was compiled with a CUDA version that is compatible with the CUDA version of the locally installed CUDA Toolkit.**

### Check Your CUDA Version

```bash
nvidia-smi
```

### Install Compatible PyTorch

Install PyTorch and torchvision for specific CUDA versions:

```bash
# CUDA version 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA version 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA version 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### CUDA Requirements

- NVIDIA GPU with CUDA capability 3.5 or higher
- CUDA toolkit matching your PyTorch version
- Compatible C++ compiler (GCC for Linux, MSVC for Windows)

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

## Troubleshooting

### CUDA Version Mismatch

If you get this error:

```
Failed to build torchlogix

...

RuntimeError:
    The detected CUDA version (11.2) mismatches the version that was used to compile
    PyTorch (11.7). Please make sure to use the same CUDA versions.
```

**Solution**: Make sure PyTorch and CUDA Toolkit versions match. Install a compatible PyTorch version or update your CUDA Toolkit.

> **Note**: Some PyTorch versions have been compiled with CUDA versions different from the advertised versions. If versions should match but don't, try other (e.g., older) PyTorch versions.

### Common Issues

**Import errors**: Ensure all dependencies are installed
```bash
pip install torch torchvision numpy scikit-learn
```

**CUDA compilation errors**: Verify CUDA toolkit installation
```bash
nvcc --version
```

**Missing dependencies**: Install development tools
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

### Disable CUDA (CPU-only)

For CPU-only installation:

```bash
export TORCHLOGIX_BUILD_CUDA_EXT=false
pip install -e .
```

## Supported Versions

- **Python**: 3.6+
- **PyTorch**: 1.9.0 - 1.13.x (tested)
- **CUDA**: 11.1 - 11.7

For experiments, install additional dependencies:
```bash
pip install -r experiments/requirements.txt
```