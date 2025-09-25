import torch

from torchlogix.models.baseline_nn import FullyConnectedNN
from torchlogix.models.conv import CNN
from torchlogix.models.nn import RandomlyConnectedNN

from .loading import input_dim_of_dataset, num_classes_of_dataset

IMPL_TO_DEVICE = {"cuda": "cuda", "python": "cpu"}

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}


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
    else:
        raise NotImplementedError(arch)

    model = model.to(llkw["device"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer
