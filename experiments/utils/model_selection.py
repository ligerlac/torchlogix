import torch

from torchlogix.models.baseline_nn import FullyConnectedNN
from torchlogix.models.conv import CNN
from torchlogix.models.nn import RandomlyConnectedNN

from torchlogix.layers import LogicConv2d, OrPooling, LogicDense, GroupSum

from .loading import input_dim_of_dataset, num_classes_of_dataset

IMPL_TO_DEVICE = {"cuda": "cuda", "python": "cpu"}

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}


llkw = {"connections": "random", "implementation": "python", "device": "cpu"}


def get_paper_model(k_num=32, tau=10, temperature=1.0, **llkw):
    """Get the model as described in the original paper."""
    # Define the model architecture
    model = torch.nn.Sequential(
        LogicConv2d(
            in_dim=28,
            num_kernels=k_num,
            channels=1,
            **llkw,
            tree_depth=3,
            receptive_field_size=5,
            padding=0,
            temperature=temperature,
        ),
        OrPooling(kernel_size=2, stride=2, padding=0),
        LogicConv2d(
            in_dim=12,
            channels=k_num,
            num_kernels=3 * k_num,
            **llkw,
            tree_depth=3,
            receptive_field_size=3,
            padding=0,
            temperature=temperature,
        ),
        OrPooling(kernel_size=2, stride=2, padding=1),
        LogicConv2d(
            in_dim=6,
            channels=3 * k_num,
            num_kernels=9 * k_num,
            **llkw,
            tree_depth=3,
            receptive_field_size=3,
            padding=0,
            temperature=temperature,
        ),
        OrPooling(kernel_size=2, stride=2, padding=1),
        torch.nn.Flatten(),
        LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, **llkw),
        LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, **llkw),
        LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, **llkw),
        GroupSum(10, tau=tau)
    )
    return model


def get_small_model(k_num=8, tau=10,**llkw):
    """Two conv blocks"""
    model = torch.nn.Sequential(
        LogicConv2d(
            in_dim=28,
            num_kernels=k_num,
            channels=1,
            **llkw,
            tree_depth=3,
            receptive_field_size=5,
            padding=0,
        ),
        OrPooling(kernel_size=2, stride=2, padding=0),
        LogicConv2d(
            in_dim=12,
            channels=k_num,
            num_kernels=3 * k_num,
            **llkw,
            tree_depth=3,
            receptive_field_size=3,
            padding=0,
        ),
        torch.nn.Flatten(),
        LogicDense(in_dim=12 * 12 * 3 * k_num, out_dim=128 * k_num, **llkw),
        GroupSum(10, tau=tau)
    )
    return model


class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x)
        return x

def get_tiny_model(k_num=4, tau=10,**llkw):
    """Get a tiny model for quick testing."""
    print(f"Using tiny model with k_num={k_num}, tau={tau}, llkw={llkw}")
    model = torch.nn.Sequential(
        LogicConv2d(
            in_dim=28,
            num_kernels=k_num,
            channels=1,
            **llkw,
            tree_depth=2,
            receptive_field_size=5,
            padding=0,
            temperature=0.2,
        ),
        OrPooling(kernel_size=2, stride=2, padding=0),
        LogicConv2d(
            in_dim=12,
            channels=k_num,
            num_kernels=3 * k_num,
            **llkw,
            tree_depth=2,
            receptive_field_size=3,
            padding=0,
            temperature=0.2,
        ),
        OrPooling(kernel_size=2, stride=2, padding=1),
        torch.nn.Flatten(),
        PrintLayer(),
        LogicDense(in_dim=6 * 6 * 3 * k_num, out_dim=128 * k_num, **llkw),
        LogicDense(in_dim=128 * k_num, out_dim=64 * k_num, **llkw),
        LogicDense(in_dim=64 * k_num, out_dim=30 * k_num, **llkw),
        GroupSum(10, tau=tau)
    )
    return model


def get_model(args):
    """
    Select model from the architecture.

    It can be a difflogic model or a baseline model.
    """
    llkw = {
        "grad_factor": args.grad_factor,
        "connections": args.connections,
        "implementation": args.implementation,
        "device": IMPL_TO_DEVICE[args.implementation],
        "parametrization": args.parametrization,
        "forward_sampling": args.forward_sampling
    }
    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)
    dtype = BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]
    arch = args.architecture
    k = args.num_neurons
    num_layers = args.num_layers
    tau = args.tau
    if arch == "randomly_connected":
        print("Using randomly connected architecture.")
        print(f"in_dim = {in_dim}, k = {k}, layers = {num_layers}")
        model = RandomlyConnectedNN(in_dim, k, num_layers, class_count, tau, **llkw)
    elif arch == "fully_connected":
        model = FullyConnectedNN(in_dim, k, num_layers, class_count, dtype)
    elif arch == "cnn":
        model = CNN(class_count, tau, **llkw)
    elif arch == "small":
        model = get_small_model(tau=tau, **llkw)
    elif arch == "tiny":
        model = get_tiny_model(tau=tau, **llkw)
    elif arch == "paper":
        model = get_paper_model(tau=tau, temperature=args.temperature, **llkw)
    else:
        raise NotImplementedError(arch)

    model = model.to(llkw["device"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer
