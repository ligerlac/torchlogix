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

**Note:** `torchlogix` is based on the `difflogic` package ([https://github.com/Felix-Petersen/difflogic/](https://github.com/Felix-Petersen/difflogic/)), and extends it by new concepts such as compact parametrizations, higher-dimensional logic blocks, learnable connections and binarization as described in "WARP Logic Neural Networks" (Paper @ [ArXiv](https://arxiv.org/abs/2602.03527)). It also implements convolutions as described in "Convolutional Differentiable Logic Gate Networks (Paper @ [ArXiv](https://arxiv.org/pdf/2411.04732)).

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
Like ordinary PyTorch neural networks, this model may be trained, e.g., with `torch.nn.CrossEntropyLoss`. The Adam optimizer with a learning rate of `0.01` works well. Every layer and hence the entire model can be switched between the relaxed trainable and discrete, fully boolean version with the standard `model.train()` / `model.eval()` commands. Furthermore, the discrete model can be expressed in pure `C` and compiled like so

```python
compiled_model = CompiledLogicNet(model, input_shape=(1, 28, 28))
compiled_model.compile()

all_preds = model(all_X)  #  ~15 ms for all 10000 test examples on my laptop
```
The full training- and evaluation of the model above is demonstrated in the example notebook [experiments/mnist_example.ipynb](experiments/mnist_example.ipynb).

There is also a preliminary functionality to generate verlilog code:
```python
verilog_code_str = compiled_model.get_verilog_code()
```

## Documentation

**More thorough documentation is available [here](https://ligerlac.github.io/torchlogix/)**, including an **API Reference**. Some quick links:
- **[Installation Guide](docs/guides/installation.md)** - Detailed installation instructions
- **[Quick Start](docs/guides/quickstart.md)** - Get started with TorchLogix in minutes
- **[Concepts](docs/guides/concepts.md)** - Understand some of the design choices behind `torchlogix`

## Experiments

Various experiments can be run using the script `experiments/train.py`. For example, the medium-sized convolutional model on CIFAR-10 from the paper  "Convolutional Differentiable Logic Gate Networks", can be trained like so:
```
python train.py --dataset cifar-10 -a ClgnCifar10Medium --connections-init-method random-unique -lr 0.02 -wd 0.002 --device cuda --compile-model
```
This achieves roughly 66% discrete test accurcay, which can be increased to 68.5% with the same architecture by learning the binarization thresholds and restricting each kernel in the first layer to just a single channel:
```
python train.py --dataset cifar-10 -a ClgnCifar10Medium2 --connections-init-method random-unique --binarization learnable -lr 0.02 -wd 0.002 --binarization-learning-rate 0.01 --device cuda --compile-model
```
The training converges within roughly 30 minutes on an `A100`. The accuracy can be increased further with data augmentation, and knowledge distillation but details of the training procedure are beyond the scope of this package.

## License

`torchlogix` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
