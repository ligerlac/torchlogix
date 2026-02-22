<!-- begin-logo -->
![torchlogix_logo](assets/logo.png)
<!-- end-logo -->

<p align="center">
  <a href="https://pypi.org/project/torchlogix/">
    <img src="https://badge.fury.io/py/torchlogix.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/ligerlac/torchlogix/actions/workflows/unit-test.yml">
    <img src="https://github.com/ligerlac/torchlogix/actions/workflows/unit-test.yml/badge.svg?branch=main" alt="Build Status">
  </a>
  <a href="https://ligerlac.github.io/torchlogix/">
    <img src="https://img.shields.io/badge/docs-online-success" alt="Documentation">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  </a>
</p>

`torchlogix` is a `PyTorch`-based library for training and inference of **logic neural networks**. These solve machine learning tasks by learning combinations of boolean logic expressions. As the choice of boolean expressions is conventionally non-differentiable, relaxations are applied to allow training with gradient-based methods. The final model can be discretized again, resulting in a fully boolean expression with extremely efficient inference, e.g., beyond a
million images of MNIST per second on a single CPU core.

**Note:** `torchlogix` is based on the `difflogic` package ([https://github.com/Felix-Petersen/difflogic/](https://github.com/Felix-Petersen/difflogic/)), and extends it by new concepts such as learnable connections, higher-dimensional logic blocks, and learnable thermometer thresholding as described in "WARP Logic Neural Networks" (Paper @ [ArXiv](https://arxiv.org/abs/2602.03527)). It also implements convolutional logic layers as described in "Convolutional Logic Gate Networks (Paper @ [ArXiv](https://arxiv.org/pdf/2411.04732)).

## Installation
```shell
pip install torchlogix                 # basic
pip install "torchlogix[dev]"          # with dev tools
```
The following software stacks have validated performance:
`python3.12` / `python3.13`, `cuda12.4` / `cuda13.0`, `torch2.6` / `torch2.9`.

## Quickstart
`torchlogix` provides learnable logic layers with `torch.nn`-like API. For example, a very simple convolutional model for MNIST can be defined like so:
```python
import torch
from torchlogix.layers import LogicDense, LogicConv2d, OrPooling2d, GroupSum, FixedBinarization

model = torch.nn.Sequential(
    # Every pixel is False (=0) or True (>0). Standard practice on MNIST
    FixedBinarization(thresholds=[0.0]),
    # Convolution w/ 16 kernels - 4 inputs each, randomly drawn from a 3x3 receptive field
    LogicConv2d(in_dim=28, channels=1, num_kernels=16, tree_depth=2, receptive_field_size=3),
    # Reduce dimensionality with pooling operation
    OrPooling2d(kernel_size=2, stride=2, padding=0),
    torch.nn.Flatten(),
    # Two randomly connected dense layers w/ 4000 neurons
    LogicDense(16*13*13, 4_000),
    LogicDense(4_000, 4_000),
    # Output 10 logits via group sum (scaled by 1/8 for smoothness)
    GroupSum(k=10, tau=8)
)
```
The model may be trained, e.g., with a `torch.nn.CrossEntropyLoss` similar to how other neural networks models are trained in PyTorch. Notably, the Adam optimizer (`torch.optim.Adam`) should be used for training and the recommended default learning rate is `0.01` instead of `0.001`.
Every layer and hence the entire model can be switched between the relaxed trainable and discrete, fully boolean version with the standard `model.train()` / `model.eval()` commands. Furthermore, the discrete model can be expressed in pure `C` and compiled like so

```python
compiled_model = CompiledLogicNet(model, input_shape=(1, 28, 28))
compiled_model.compile()

all_preds = model(all_X)  #  ~15 ms for all 10000 test examples 
```
The full training- and evaluation of the model above is demonstrated in the example notebook [experiments/mnist_example.ipynb](experiments/mnist_example.ipynb).

## Documentation

**Full documentation is available [here](https://ligerlac.github.io/torchlogix/)**, including a full **API Reference**. Some quick links:
- **[Installation Guide](docs/guides/installation.md)** - Detailed installation instructions
- **[Quick Start](docs/guides/quickstart.md)** - Get started with TorchLogix in minutes
- **[Concepts](docs/guides/concepts.md)** - Understand some of the design choices behind `torchlogix`

## 🧪 Experiments

There are experiments on CIFAR-10 in the `experiments` directory. We will add more soon.

## 📜 License

`torchlogix` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.

# TODO
- [] revert .gitignore
- [] rework evaluate.py and compile.py
- [] Move all helper functions, except for run_training in utils.py?
- [x] Describe MNIST example
