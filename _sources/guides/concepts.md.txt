# Core Concepts

This guide explains the key architectural concepts and design principles behind TorchLogix. For detailed API documentation, see the [API Reference](../api/torchlogix.rst).

## Design Philosophy

TorchLogix is built around three core principles:

1. **PyTorch API Compatibility**: Layers follow standard `torch.nn.Module` conventions
2. **Separation of Concerns**: Each layer separates connection patterns from parametrization strategies
3. **Differentiable-to-Discrete**: Training uses continuous relaxations with multiple discretization strategies

---

## PyTorch API Compatibility

All TorchLogix layers follow standard PyTorch `nn.Module` conventions, making them compatible with existing PyTorch workflows. However, **logic layers are not drop-in replacements** for standard layers like `nn.Linear` or `nn.Conv2d` - they have fundamentally different computational properties and require significantly more neurons (or kernels) to achieve comparable expressiveness.

```python
import torch.nn as nn
from torchlogix.layers import LogicDense

# Standard PyTorch: 256 neurons
linear = nn.Linear(784, 256)

# TorchLogix: Need 1000+ neurons for similar capacity
logic = LogicDense(784, 4096)  # More neurons required!
```

### Standard PyTorch Conventions

Despite different computational characteristics, TorchLogix layers follow all standard PyTorch patterns:

- **Inherit from `nn.Module`**: All layers are proper PyTorch modules
- **Learnable parameters**: Registered via `nn.Parameter` for automatic gradient computation
- **Training/eval modes**: `.train()` and `.eval()` control behavior
- **Device management**: Standard `.to(device)` and `.cuda()` work as expected
- **Composition**: Use with `nn.Sequential`, custom `forward()` methods, etc.
- **State dict**: Save and load with `state_dict()` and `load_state_dict()`

This means you can use TorchLogix with standard PyTorch optimizers, learning rate schedulers, data loaders, and training loops - just be mindful that logic networks need different architectural scales (and typically much higher learning rates).

---

## Separation of Concerns

Each logic layer is composed of two independent, swappable components:

```
┌─────────────────────────────────────────┐
│          Logic Layer (nn.Module)        │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Connections │  │ Parametrization  │  │
│  │             │  │                  │  │
│  │ Which       │  │ - How Boolean    │  │
│  │ inputs      │  │   functions are  │  │
│  │ connect?    │  │   represented?   │  │
│  │             │  │ - Forward        │  │
│  │             │  │   sampling mode  │  │
│  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
```

This modular design allows you to mix and match different strategies for each component.

### 1. Connections: Input Routing

The **Connections** component determines which inputs from the previous layer connect to each neuron or convolutional kernel.

**Why separate this?** Different connectivity patterns enable different trade-offs between expressiveness and parameter efficiency.

#### Types of Connections

**Fixed Connections** (`connections="fixed"`):
- Randomly select which inputs connect to each neuron
- Fixed after initialization (no learning overhead)
- Options: `"random"` (with replacement) or `"random-unique"` (without replacement)
- Used in most pre-configured models for efficiency

**Learnable Connections** (`connections="learnable"`):
- Learn which inputs to connect during training
- Uses Gumbel-Softmax for differentiable selection
- More flexible but adds computational cost
- Useful when input structure is unknown

#### Dense vs Convolutional Structure

**Dense Structure**:
- Each neuron selects `lut_rank` inputs from all previous layer outputs
- Fully connected within the selection

**Convolutional Structure**:
- Each kernel selects inputs from a spatial receptive field
- Organized as a binary tree with depth `tree_depth`
- Supports channel grouping via `channel_group_size`
- By default, channels balanced. E.g. a kernel w/ 8 inputs and and a channel group size of 2 would pick 4 inputs from each channel

### 2. Parametrization: Representing Boolean Functions

The **Parametrization** component defines how Boolean functions (look-up tables) are represented as learnable parameters and how they are sampled during forward passes.

**Why separate this?** Different parametrizations have different optimization landscapes and computational costs, and different sampling strategies enable different exploration-exploitation trade-offs.

Any Boolean function can be represented as a weighted sum over a basis. TorchLogix provides three parametrization strategies:

#### Raw Parametrization (`parametrization="raw"`)

Directly represents the truth table using all 16 possible 2-input Boolean functions as described in https://arxiv.org/abs/2210.08277

- **Weights**: 16 logits per neuron (one per Boolean function)
- **Sampling**: Softmax over the 16 functions
- **Limitation**: Only supports `lut_rank=2` (2 inputs)
- **Use case**: Baseline method, interpretable gate selection

```
Each neuron learns: [w₀, w₁, ..., w₁₅]
                      ↓
                   softmax
                      ↓
       Weighted sum of 16 Boolean gates
```

#### WARP Parametrization (`parametrization="warp"`)

Represents functions using Walsh-Hadamard basis coefficients (parity functions) as described in https://arxiv.org/abs/2602.03527

- **Weights**: 2^k coefficients per neuron (where k = `lut_rank`)
- **Sampling**: Sigmoid-based thresholding
- **Supports**: `lut_rank` ∈ {1, 2, 4, 6}
- **Use case**: Fewer parameters than raw, scales to higher-rank LUTs

```
Walsh basis: {1, x₁, x₂, x₁⊕x₂}  (for k=2)

Each neuron learns: [c₀, c₁, c₂, c₃]
                      ↓
          f(x₁,x₂) = Σᵢ cᵢ · basisᵢ(x₁,x₂)
```

**Benefits**:
- More parameter-efficient than raw (4 coefficients vs 16 for `lut_rank=2`)
- Scales to higher-rank LUTs (4-input, 6-input gates)

#### Light Parametrization (`parametrization="light"`)

Uses indicator polynomial basis (product terms) with all-positive coefficients as described in https://arxiv.org/abs/2510.03250

- **Weights**: 2^k sigmoid-mapped coefficients
- **Supports**: `lut_rank` ∈ {2, 4, 6}
- **Use case**: Alternative basis with different inductive bias

```
Light basis: {1, x₁, x₂, x₁·x₂}  (for k=2)

Each neuron learns: [w₀, w₁, w₂, w₃]
                      ↓
                  sigmoid(wᵢ) → positive
                      ↓
          f(x₁,x₂) = Σᵢ σ(wᵢ) · basisᵢ(x₁,x₂)
```

#### LUT Rank: Higher-Order Logic

The `lut_rank` parameter controls how many inputs each logic gate operates on. There are `2^(2^n)` boolean functions w/ `n` inputs:

- **`lut_rank=2`**: 2-input gates (16 possible functions)
- **`lut_rank=4`**: 4-input gates (65,536 possible functions)
- **`lut_rank=6`**: 6-input gates (18 quintillion possible functions)

Higher rank = more expressive gates but exponentially more parameters.

#### Weight Initialization

The `weight_init` parameter controls how parameters are initialized:

- **`"residual"`** (default): Initialize near identity/passthrough
  - Critical for training deep networks (>6 layers)
  - Prevents vanishing gradients at initialization
  - Typically results in more trivial identity gates are optimized away at compile-time

- **`"random"`**: Random initialization
  - Good for shallow networks
  - May struggle with depth >6

- **`"residual-catalog"`**: Mix of identity and random (WARP only)
  - Used in WARP-LUTs paper experiments

#### Forward Sampling Modes

The `forward_sampling` parameter (part of parametrization) controls how continuous relaxations are converted to outputs during the forward pass.

| Mode | Description | When to Use |
|------|-------------|-------------|
| **`"soft"`** | Continuous softmax/sigmoid relaxation | Default, stable gradients |
| **`"hard"`** | Straight-through estimator (STE) | Reduce train-test mismatch |
| **`"gumbel_soft"`** | Gumbel-Softmax/Sigmoid with noise | Exploration during training |
| **`"gumbel_hard"`** | Gumbel + STE | Exploration + discretization |


**Straight-Through Estimators (STE)**:

Hard sampling modes use the straight-through estimator trick:
- **Forward pass**: Discrete (argmax or threshold)
- **Backward pass**: Gradient of continuous relaxation

This allows training discrete models with gradient-based optimization:

```
Forward:  y = argmax(logits)           [discrete, non-differentiable]
Backward: ∂L/∂logits = ∂L/∂softmax    [continuous, differentiable]
```

**Gumbel Noise Injection**:

Gumbel sampling modes add noise to enable exploration:

```python
# Without Gumbel: deterministic selection
logits = [2.0, 1.0, 0.5, 0.1]
probs = softmax(logits)  # Always picks first

# With Gumbel: stochastic selection
logits = [2.0, 1.0, 0.5, 0.1]
noisy_logits = logits + Gumbel(0, 1)
probs = softmax(noisy_logits / temperature)  # Explores
```

**Training vs Evaluation Modes**:

The sampling mode only applies during training. At evaluation time, the model always discretizes:

```python
# Training mode
model.train()
output = model(x)  # Uses specified forward_sampling mode

# Evaluation mode
model.eval()
output = model(x)  # Always uses hard discretization (argmax/threshold)
```

This ensures that:
1. Training benefits from gradient flow through continuous relaxations
2. Inference uses fully discrete operations (fast, interpretable)

---

## Layer-Specific Concepts

### LogicDense: Fully-Connected Layers

`LogicDense` is the fundamental building block, analogous to `nn.Linear`.

**Key differences from `nn.Linear`**:
- No bias term (bias is a Boolean function: constant True/False)
- Neurons operate on `lut_rank` inputs (not all inputs)
- Output is bounded [0, 1] in eval mode (Boolean values)
- Requires many more neurons for comparable expressiveness

**Computation flow**:
```
Input x: (batch, in_dim)
   ↓
Connections: select lut_rank inputs per neuron
   ↓
Shape: (batch, lut_rank, out_dim)
   ↓
Parametrization: apply Boolean functions
   ↓
Output: (batch, out_dim)
```

### LogicConv2d/3d: Convolutional Layers

`LogicConv2d` and `LogicConv3d` are analogous to `nn.Conv2d` and `nn.Conv3d`.

**Key architectural difference**: Uses a **binary tree** of logic gates instead of a single large LUT.

#### Binary Tree Structure

Each convolutional kernel is organized as a binary tree:

```
                    Output
                      ↑
                  Level 2: 1 gate
                    ↗   ↖
              Level 1: 2 gates
              ↗  ↖      ↗  ↖
         Level 0: 4 positions in receptive field
```

- **Tree depth**: Controlled by `tree_depth` parameter
- **Receptive field**: `lut_rank^tree_depth` spatial positions
- **Example**: `lut_rank=2`, `tree_depth=2` → 4 positions in receptive field

**Why use a tree?** Enables hierarchical composition of local features without exponential parameter growth.

**Weight structure**: Each tree level has separate learnable parameters:
```python
# LogicConv2d with tree_depth=2
self.weight = nn.ParameterList([
    nn.Parameter(...),  # Level 0: 4 kernels
    nn.Parameter(...),  # Level 1: 2 kernels
    nn.Parameter(...)   # Level 2: 1 kernel (output)
])
```

#### Channel Grouping

The `channel_group_size` parameter restricts each kernel to a subset of input channels:

```python
layer = LogicConv2d(
    in_channels=32,
    num_kernels=64,
    channel_group_size=8  # Each kernel sees 8 input channels
)
```

This creates overlapping groups, reducing parameters while maintaining coverage.

---

## Supporting Components

### GroupSum: Classification Head

`GroupSum` aggregates neuron outputs into class logits:

```python
# Input: (batch, n_neurons)
# Output: (batch, num_classes=:k)

layer = GroupSum(
    k=10,
    tau=1.0,  # Temperature for normalization
    beta=0.0  # Offset, useful for regression tasks
)
```

**How it works**:
1. Reshape `(batch, num_classes × neurons_per_class)` → `(batch, num_classes, neurons_per_class)`
2. Sum over neuron groups
3. Divide by `tau` to normalize range
4. Shift by `beta` to desired range

**Why `tau`?** Each Boolean neuron outputs [0, 1], so a sum of 100 neurons is [0, 100]. Setting `tau=100` normalizes to [0, 1] range. And setting `beta` to `-42` would result in [-42, -41].

### Binarization Layers

Since logic gates operate on Boolean values, TorchLogix provides layers to convert continuous inputs:

- **`FixedBinarization`**: Fixed thresholds (e.g., 0.5)
- **`LearnableBinarization`**: Learn thresholds during training
- **`SoftBinarization`**: Differentiable sigmoid-based thresholding, but with fixed position. Can aid training but temperature should be annealed during training to close discretization gap.

Inputs can be binarized `per_feature`, or `global`. For image-like data, there is also the `per_channel` option.

---

## Advanced Topics

### Gradient Scaling for Deep Networks

The `grad_factor` parameter scales gradients during the backward pass:

```python
layer = LogicDense(
    in_dim=256,
    out_dim=256,
    grad_factor=2.0  # Scale gradients by 2x
)
```

**Why needed?** Logic gates can have very small gradients (e.g., AND gate: `∂(a·b)/∂a = b`). For deep networks (>6 layers), this causes vanishing gradients.

**How it works**: Uses a custom autograd function:
- **Forward pass**: Identity (`y = x`)
- **Backward pass**: Scale gradient (`∂L/∂x = grad_factor · ∂L/∂y`)

**When to use**: Set `grad_factor=2.0` or higher for networks deeper than 6 layers.

### Extracting Discrete Boolean Functions

After training, you can extract the learned Boolean functions:

```python
model.eval()  # Important: switch to eval mode

# Get truth tables
luts = model.get_luts()  # Shape: (num_neurons, 2^lut_rank)

# Get integer IDs (for lut_rank=2 only)
luts, ids = model.get_luts_and_ids()  # ids in [0, 15]
```

The `ids` map to the 16 Boolean functions (0=False, 1=AND, 7=OR, 14=NAND, 15=True).

---

## Deployment: From Training to Inference

TorchLogix provides multiple inference backends optimized for different use cases:

### 1. Standard PyTorch (Training & Validation)

```python
model = LogicDense(128, 64)
model.eval()  # Discretizes to Boolean operations
output = model(input)  # Standard PyTorch inference
```

**Use case**: Validation during training, small-scale inference.

### 2. PackBitsTensor (GPU Batch Inference)

```python
from torchlogix import PackBitsTensor

model.eval()
model.cuda()

# Pack boolean tensors to single bits
input_packed = PackBitsTensor.pack(input)  # Requires CUDA
output_packed = model(input_packed)
output = output_packed.unpack()
```

**Use case**: Large batch inference on GPU, up to 32× memory savings.

**Limitation**: Requires CUDA device.

### 3. CompiledLogicNet (CPU Production Inference)

```python
from torchlogix import CompiledLogicNet

# Compile to C code
compiled = CompiledLogicNet(model, compiler='gcc', optimization_level='-O3')

# Run inference
output = compiled(input)  # Calls compiled C code
```

**Use case**: Production deployment on CPU, edge devices.

**Benefits**: 10-100× faster than PyTorch, no Python/PyTorch dependencies at runtime.

---

## Common Patterns

### Building a Classifier

```python
import torch.nn as nn
from torchlogix.layers import LogicDense, GroupSum, LearnableBinarization

model = nn.Sequential(
    LearnableBinarization(num_thresholds=3),  # Input preprocessing
    nn.Flatten(),
    LogicDense(in_dim=28*28*3, out_dim=512),
    LogicDense(in_dim=512, out_dim=512),
    LogicDense(in_dim=512, out_dim=1000),
    GroupSum(num_classes=10, neurons_per_class=100, tau=100)
)
```

### Using Pre-configured Models

```python
from torchlogix.models import DlgnMediumMnist, ClgnMediumCifar10

# Dense model for MNIST
model = DlgnMediumMnist()

# Convolutional model for CIFAR-10
model = ClgnMediumCifar10()
```

Pre-configured models follow naming convention: `{D/C}lgn{Size}{Dataset}{Variant}`
- `D` = Dense, `C` = Convolutional
- Size: Tiny, Small, Medium, Large, Large2, Large4
- Dataset: Mnist, Cifar10
- Variant: (empty), Rank4, Rank6, Learn0, Learn1, Learn2

---

## Summary

TorchLogix's architecture is built on three key ideas:

1. **PyTorch Compatibility**: Seamless integration with existing PyTorch workflows (but not drop-in replacements)
2. **Modular Design**: Separate concerns (connections, parametrization) for maximum flexibility
3. **Differentiable-to-Discrete**: Train with continuous relaxations, deploy with discrete operations

This design allows you to:
- Use logic layers like any other PyTorch layer
- Mix and match different connection patterns, parametrization strategies, and sampling modes
- Train with gradients and deploy as efficient Boolean circuits

For detailed API documentation, see the [API Reference](../api/torchlogix.rst).

For hands-on examples, see the [Quickstart Guide](quickstart.md).
