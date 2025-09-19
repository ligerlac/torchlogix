# Logic Gates in TorchLogix

## Overview

TorchLogix implements neural networks using differentiable logic gate operations. Instead of traditional linear transformations followed by nonlinear activations, logic layers combine inputs using the 16 possible binary Boolean operations.

## The 16 Boolean Operations

Each logic gate is identified by an ID (0-15) and implements one of the complete set of binary Boolean functions:

| ID | Operation | Name | AB=00 | AB=01 | AB=10 | AB=11 | Formula |
|----|-----------|------|-------|-------|-------|-------|---------|
| 0  | False     | ZERO | 0     | 0     | 0     | 0     | 0 |
| 1  | AND       | AND  | 0     | 0     | 0     | 1     | A ∧ B |
| 2  | A and not B | -   | 0     | 0     | 1     | 0     | A ∧ ¬B |
| 3  | A         | BUFA | 0     | 0     | 1     | 1     | A |
| 4  | B and not A | -   | 0     | 1     | 0     | 0     | ¬A ∧ B |
| 5  | B         | BUFB | 0     | 1     | 0     | 1     | B |
| 6  | XOR       | XOR  | 0     | 1     | 1     | 0     | A ⊕ B |
| 7  | OR        | OR   | 0     | 1     | 1     | 1     | A ∨ B |
| 8  | NOR       | NOR  | 1     | 0     | 0     | 0     | ¬(A ∨ B) |
| 9  | XNOR      | XNOR | 1     | 0     | 0     | 1     | ¬(A ⊕ B) |
| 10 | NOT B     | NOTB | 1     | 0     | 1     | 0     | ¬B |
| 11 | B implies A | -   | 1     | 0     | 1     | 1     | B → A |
| 12 | NOT A     | NOTA | 1     | 1     | 0     | 0     | ¬A |
| 13 | A implies B | -   | 1     | 1     | 0     | 1     | A → B |
| 14 | NAND      | NAND | 1     | 1     | 1     | 0     | ¬(A ∧ B) |
| 15 | True      | ONE  | 1     | 1     | 1     | 1     | 1 |

## Differentiable Implementation

In traditional Boolean logic, operations work with discrete {0, 1} values. TorchLogix extends this to continuous [0, 1] values, making the operations differentiable:

```python
# Traditional Boolean AND
result = a & b

# Differentiable AND in TorchLogix
result = a * b
```

### Key Operations

**AND (ID=1)**: `a * b`
- Multiplicative combination
- Output is high only when both inputs are high

**OR (ID=7)**: `a + b - a * b`
- Additive combination with correction term
- Output is high when either input is high

**XOR (ID=6)**: `a + b - 2 * a * b`
- Exclusive or operation
- Output is high when inputs differ

**NAND (ID=14)**: `1 - a * b`
- Negated AND
- Universal gate in Boolean logic

## Binary Tree Structure

Logic layers organize operations in binary trees:

```
Level 0 (inputs):    a₁  a₂  a₃  a₄
                      ↓   ↓   ↓   ↓
Level 1:           gate₁   gate₂
                      ↓       ↓
Level 2:           final_gate
                       ↓
Output:              result
```

### Tree Depth

- **Depth 1**: Single operation combining two inputs
- **Depth 2**: 3 operations (2 at level 1, 1 at level 2)
- **Depth 3**: 7 operations (4 + 2 + 1)
- **Depth n**: 2ⁿ - 1 total operations

## Learning Process

During training, the network learns:

1. **Which operations to use**: Soft selection over all 16 operations
2. **How to connect inputs**: Random or learned connectivity patterns
3. **Operation weights**: Continuous values that evolve during training

### Soft Gate Selection

Instead of hard selection, TorchLogix uses soft weights:

```python
# Weighted combination of all operations
result = Σᵢ wᵢ * operation_i(a, b)

# Where weights are softmax-normalized
w = softmax(learned_parameters)
```

## Practical Considerations

### Expressiveness
- Any Boolean function can be expressed with sufficient depth
- NAND and NOR are universal gates
- Complex functions require deeper trees

### Training Dynamics
- Start with random operation weights
- Gradually specialize to useful combinations
- Gradient flow through differentiable operations

### Computational Efficiency
- Parallel evaluation of operations
- CUDA acceleration available
- Compiled models for inference

## Examples

### Simple Logic Functions

```python
# Learn XOR function
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
targets = torch.tensor([0, 1, 1, 0], dtype=torch.float32)

layer = LogicDense(in_dim=2, out_dim=1, tree_depth=2)
# Train to learn XOR...
```

### Complex Pattern Recognition

Logic gates can learn complex patterns by combining simple operations hierarchically, making them suitable for tasks like image classification and sequence modeling.