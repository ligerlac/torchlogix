<!-- begin-logo -->
![torchlogix_logo](assets/logo.png)
<!-- end-logo -->

[![PyPI version](https://badge.fury.io/py/torchlogix.svg)](https://pypi.org/project/torchlogix/)
[![Python 3.10‒3.13](https://img.shields.io/badge/python-3.10%E2%80%923.14-blue)](https://www.python.org)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Test](https://github.com/ligerlac/torchlogix/actions/workflows/unit-test.yml/badge.svg?branch=main)](https://github.com/ligerlac/torchlogix/actions/workflows/unit-test.yml)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://ligerlac.github.io/torchlogix/)

`torchlogix` is a `PyTorch`-based library for training and inference of **logic neural networks**. These solve machine learning tasks by learning combinations of boolean logic expressions. As the choice of boolean expressions is conventionally non-differentiable, relaxations are applied to allow training with gradient-based methods. The final model can be discretized again, resulting in a fully boolean expression with extremely efficient inference, e.g., beyond a
million images of MNIST per second on a single CPU core.

**Note:** `torchlogix` is based on the `difflogic` package ([https://github.com/Felix-Petersen/difflogic/](https://github.com/Felix-Petersen/difflogic/)), and extends it by new concepts such as learnable connections, higher-dimensional logic expressions, and learnable thermometer thresholding as described in "WARP Logic Neural Networks" (Paper @ [ArXiv](https://arxiv.org/abs/2602.03527)). It also implements convolutional logic layers as described in the Paper "Convolutional Logic Gate Networks (Paper @ [ArXiv](https://arxiv.org/pdf/2411.04732)).

## Installation
```shell
pip install torchlogix                 # basic
pip install "torchlogix[dev]"          # with dev tools
```
The following software stacks have validated performance:
`python3.12` / `python3.13`, `cuda12.4` / `cuda13.0`, `torch2.6` / `torch2.9`.

## 📚 Documentation

**Full documentation is available [here](https://ligerlac.github.io/torchlogix/)**, including a full **API Reference**. Some quick links:
- **[Installation Guide](docs/guides/installation.md)** - Detailed installation instructions
- **[Quick Start](docs/guides/quickstart.md)** - Get started with TorchLogix in minutes
- **[Concepts](docs/guides/concepts.md)** - Understand some of the design choices behind `torchlogix`
## 🌱 Intro and Training

This library provides a framework for both training and inference with logic gate networks.
The following gives an example of a definition of a differentiable logic network model for the MNIST data set:

```python
import torch
from torchlogix.layers import LogicDense, LogicConv2d, OrPooling, GroupSum, LearnableThermometerThresholding

model = torch.nn.Sequential(
    LogicConv2d(in_dim=28, num_kernels=64, receptive_field_size=5),
    OrPooling(kernel_size=2, stride=2, padding=0),
    LogicConv2d(in_dim=12, num_kernels=256, receptive_field_size=3),
    torch.nn.Flatten(),
    LogicLayer(256*10*10, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    GroupSum(k=10, tau=30)
)
```

This model receives a `(1,28,28)` dimensional input and returns `k=10` values corresponding to the 10 classes of MNIST.
The model may be trained, e.g., with a `torch.nn.CrossEntropyLoss` similar to how other neural networks models are trained in PyTorch.
Notably, the Adam optimizer (`torch.optim.Adam`) should be used for training and the recommended default learning rate is `0.01` instead of `0.001`.
Finally, it is also important to note that the number of neurons in each layer is much higher for logic gate networks compared to
conventional MLP neural networks because logic gate networks are very sparse.

To go into details, for each of these modules, in the following we provide more in-depth examples:

```python
layer = DenseLogic(
    in_dim=784,               # number of inputs
    out_dim=16_000,           # number of outputs
    device='cuda',            # the device (cuda / cpu)
    connections='random',     # the method for the initialization of the connections
    parametrization='raw',    # classic 16 weights per node (one per gate) one of two 4-weight parametrizations ('anf' or 'walsh')
    weight_init="residual",   # weight initialization scheme ("random" or "residual")
    forward_sampling="soft"   # Method for the foward pass: "soft", "hard", "gumbel_soft", or "gumbel_hard"
)
layer = LogicConv2d(
    in_dim=28,               # dimension of input (can be two-tuple for non-quadratic shapes)
    channels=3,              # number of channels of the input (1 for grey-scale)
    num_kernels=32,          # number of convolutional kernels (filters)
    tree_depth=3,            # depth of the binary logic tree that make up each kernel
    receptive_field_size=3,  # comparable to kernel size in ordinary convolutional kernels (can be two-tuple for non-quadratic shapes)
    padding=0,
    ... # all other keyword arguments like dense layer above
)
```

At this point, it is important to discuss the `device` option. `torchlogix` is implemented in pure PyTorch and works with both `device='cpu'` and `device='cuda'`. For best performance, use PyTorch's built-in optimizations like `torch.compile`.

To aggregate output neurons into a lower dimensional output space, we can use `GroupSum`, which aggregates a number of output neurons into
a `k` dimensional output, e.g., `k=10` for a 10-dimensional classification setting.
It is important to set the parameter `tau`, which the sum of neurons is divided by to keep the range reasonable.
As each neuron has a value between 0 and 1 (or in inference a value of 0 or 1), assuming `n` output neurons of the last `LogicLayer`,
the range of outputs is `[0, n / k / tau]`.

## 🖥 Model Inference

During training, the model should remain in the PyTorch training mode (`.train()`), which keeps the model differentiable.
However, we can easily switch the model to a hard / discrete / non-differentiable model by calling `model.eval()`, i.e., for inference.
Typically, this will simply discretize the model but not make it faster per se.

However, there are two modes that allow for fast inference:

### `PackBitsTensor`

The first option is to use a `PackBitsTensor`.
`PackBitsTensor`s allow efficient dynamic execution of trained logic gate networks on GPU.

A `PackBitsTensor` can package a tensor (of shape `b x n`) with boolean
data type in a way such that each boolean entry requires only a single bit (in contrast to the full byte typically
required by a bool) by packing the bits along the batch dimension. If we choose to pack the bits into the `int32` data
type (the options are 8, 16, 32, and 64 bits), we would receive a tensor of shape `ceil(b/32) x n` of dtype `int32`.
To create a `PackBitsTensor` from a boolean tensor `data`, simply call:
```python
data_bits = torchlogix.PackBitsTensor(data)
```
To apply a model to the `PackBitsTensor`, simply call:
```python
output = model(data_bits)
```
This requires that the `model` is in `.eval()` mode, and if supplied with a `PackBitsTensor`, will automatically use
a logic gate-based inference on the tensor. Note that `PackBitsTensor` requires a CUDA-enabled device.
It is notable that, while the model is in `.eval()` mode, we can still also feed float tensors through the model, in
which case it will simply use a hard variant of the real-valued logics.

### `CompiledLogicNet`

The second option is to use a `CompiledLogicNet`.
This allows especially efficient static execution of a fixed trained logic gate network on CPU.
Specifically, `CompiledLogicNet` converts a model into efficient C code and can compile this code into a binary that
can then be efficiently run or exported for applications.
The following is an example for creating `CompiledLogicNet` from a trained `model`:

```python
compiled_model = torchlogix.CompiledLogicNet(
    model=model,            # the trained model (should be a `torch.nn.Sequential` with `LogicLayer`s)
    num_bits=64,            # the number of bits of the datatype used for inference (typically 64 is fastest, should not be larger than batch size)
    cpu_compiler='gcc',     # the compiler to use for the c code (alternative: clang)
    verbose=True
)
compiled_model.compile(
    save_lib_path='my_model_binary.so',  # the (optional) location for storing the binary such that it can be reused
    verbose=True
)

# to apply the model, we need a 2d numpy array of dtype bool, e.g., via  `data = data.bool().numpy()`
output = compiled_model(data)
```

This will compile a model into a shared object binary, which is then automatically imported.
To export this to other applications, one may either call the shared object binary from another program or export
the model into C code via `compiled_model.get_c_code()`.
A limitation of the current `CompiledLogicNet` is that the compilation time can become long for large models.

We note that between publishing the paper and the publication of `torchlogix`, we have substantially improved the implementations.
Thus, the model inference modes have some deviation from the implementations for the original paper as we have
focussed on making it more scalable, efficient, and easier to apply in applications.
We have especially focussed on modularity and efficiency for larger models and have opted to polish the presented
implementations over publishing a plethora of different competing implementations.

We experimented with custom CUDA kernels for acceleration but found that we were unable to beat `torch.compile`'s performance.
As a result, the library uses pure PyTorch implementations that benefit from PyTorch's built-in optimizations.

## 🧪 Experiments

There are experiments on CIFAR-10 in the `experiments` directory. We will add more soon.

## 📜 License

`torchlogix` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.

# TODO
- [] revert .gitignore
- [] rework evaluate.py
- [] Move all helper functions, except for run_training in utils.py?
