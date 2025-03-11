import torch
import math


class BinaryLinear:
    def __init__(self, in_features, out_features, bias=True):
        """
        Custom implementation of a linear layer.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool): Whether to include a bias term
        """
        # Initialize weights using Kaiming/He initialization
        # (good for ReLU activations)
        self.weight = torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features)
        self.use_bias = bias

        # Initialize bias if used
        if bias:
            # Initialize bias to zeros
            self.bias = torch.zeros(out_features)
        else:
            self.bias = None

        # For automatic differentiation
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def __call__(self, x):
        """
        Forward pass of the linear layer.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return self.forward(x)

    def forward(self, x):
        """
        Forward pass implementation: y = xW^T + b

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Matrix multiplication: y = xW^T
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)
        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x


        output = torch.matmul(x, self.weight.t())

        # Add bias if used: y = xW^T + b
        if self.use_bias:
            output = output + self.bias

        return output

    def parameters(self):
        """
        Return the parameters of the layer for optimization.

        Returns:
            List of parameter tensors
        """
        if self.use_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __repr__(self):
        """
        String representation of the layer.

        Returns:
            String describing the layer
        """
        return f"CustomLinear(in_features={self.weight.size(1)}, out_features={self.weight.size(0)}, bias={self.use_bias})"


# Example usage
if __name__ == "__main__":
    # Create a custom linear layer with 10 input features and 5 output features
    linear = CustomLinear(in_features=10, out_features=5, bias=True)

    # Create a batch of 3 samples, each with 10 features
    x = torch.randn(3, 10)

    # Perform forward pass
    y = linear(x)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Weight shape: {linear.weight.shape}")
    if linear.bias is not None:
        print(f"Bias shape: {linear.bias.shape}")

    # Verify that this is equivalent to a PyTorch Linear layer
    pytorch_linear = torch.nn.Linear(10, 5)

    # Set the PyTorch Linear layer's parameters to match our custom layer
    with torch.no_grad():
        pytorch_linear.weight.copy_(linear.weight)
        pytorch_linear.bias.copy_(linear.bias)

    # Compare outputs
    custom_output = linear(x)
    pytorch_output = pytorch_linear(x)
    difference = torch.abs(custom_output - pytorch_output).sum().item()

    print(f"Difference between custom and PyTorch implementation: {difference}")