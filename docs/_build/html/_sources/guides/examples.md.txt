# Examples

## MNIST Classification

Complete example for training on MNIST dataset:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchlogix.models import CNN

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model setup
model = CNN(class_count=10, tau=1.0, device='cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

    print(f'Epoch {epoch} completed, Average Loss: {total_loss/len(train_loader):.6f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

## Custom Logic Network

Building a custom architecture:

```python
import torch
import torch.nn as nn
from torchlogix.layers import LogicConv2d, LogicDense, OrPooling, GroupSum

class CustomLogicNet(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.1):
        super().__init__()

        # Feature extraction
        self.conv1 = LogicConv2d(
            in_dim=input_size,
            channels=1,
            num_kernels=32,
            tree_depth=3,
            receptive_field_size=3,
            padding=1,
            connections="random-unique"
        )
        self.pool1 = OrPooling(kernel_size=2, stride=2, padding=0)

        self.conv2 = LogicConv2d(
            in_dim=(input_size[0]//2, input_size[1]//2),
            channels=32,
            num_kernels=64,
            tree_depth=4,
            receptive_field_size=3,
            padding=1
        )
        self.pool2 = OrPooling(kernel_size=2, stride=2, padding=0)

        # Classification head
        conv_output_size = 64 * (input_size[0]//4) * (input_size[1]//4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            LogicDense(in_dim=conv_output_size, out_dim=512, tree_depth=5),
            nn.Dropout(dropout),
            LogicDense(in_dim=512, out_dim=256, tree_depth=4),
            GroupSum(num_classes, tau=1.0)
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.classifier(x)
        return x

# Usage
model = CustomLogicNet(input_size=(28, 28), num_classes=10)
```

## Model Compilation and Optimization

Converting trained models for efficient inference:

```python
from torchlogix import CompiledLogicNet

# After training your model
trained_model = CNN(class_count=10, tau=1.0)
# ... training code ...

# Compile for efficient inference
compiled_model = CompiledLogicNet(
    model=trained_model,
    input_shape=(1, 28, 28),
    device='cuda',
    verbose=True  # Show compilation progress
)

# Benchmark comparison
import time

test_input = torch.randn(100, 1, 28, 28).cuda()

# Original model
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = trained_model(test_input)
original_time = time.time() - start_time

# Compiled model
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = compiled_model(test_input)
compiled_time = time.time() - start_time

print(f"Original model: {original_time:.4f}s")
print(f"Compiled model: {compiled_time:.4f}s")
print(f"Speedup: {original_time/compiled_time:.2f}x")
```

## Advanced Training Techniques

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Adaptive scheduling based on validation loss
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

for epoch in range(epochs):
    # Training...

    # Validation
    val_loss = validate(model, val_loader)

    # Update learning rate
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For StepLR
```

### Custom Loss Functions

```python
class LogicRegularizedLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, model):
        # Standard classification loss
        ce = self.ce_loss(outputs, targets)

        # Regularization: encourage diverse gate usage
        reg_loss = 0
        for module in model.modules():
            if hasattr(module, 'tree_weights'):
                for level_weights in module.tree_weights:
                    for weights in level_weights:
                        # Entropy regularization
                        probs = torch.softmax(weights, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum()
                        reg_loss += -entropy  # Negative because we want high entropy

        return ce + self.alpha * reg_loss

# Usage
criterion = LogicRegularizedLoss(alpha=0.01)
loss = criterion(outputs, targets, model)
```

## Debugging and Visualization

### Inspecting Learned Operations

```python
def analyze_logic_operations(model):
    """Analyze which logic operations the model has learned."""

    for name, module in model.named_modules():
        if hasattr(module, 'tree_weights'):
            print(f"\nLayer: {name}")

            for level_idx, level_weights in enumerate(module.tree_weights):
                print(f"  Level {level_idx}:")

                for node_idx, weights in enumerate(level_weights):
                    # Get the most likely operation for each kernel
                    probs = torch.softmax(weights, dim=-1)
                    top_ops = torch.argmax(probs, dim=-1)

                    print(f"    Node {node_idx}: Operations {top_ops.tolist()}")

# Analyze trained model
analyze_logic_operations(trained_model)
```

### Performance Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.profiler.record_function("model_inference"):
        output = model(test_input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```