import torch

from neurodifflogic.models.baseline_nn import FullyConnectedNN
from neurodifflogic.models.conv import CNN
from neurodifflogic.models.gat_conv import GATBaseline, GATDifflog
from neurodifflogic.models.gcn_conv import GCN, GCNCoraBaseline
from neurodifflogic.models.nn import RandomlyConnectedNN
from neurodifflogic.utils.loading import input_dim_of_dataset, num_classes_of_dataset

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
    device = IMPL_TO_DEVICE[args.implementation]
    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)
    dtype = BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]
    arch = args.architecture
    k = args.num_neurons
    num_layers = args.num_layers
    tau = args.tau
    if arch == "randomly_connected":
        model = RandomlyConnectedNN(in_dim, k, num_layers, class_count, tau, **llkw)
    elif arch == "fully_connected":
        model = FullyConnectedNN(in_dim, k, num_layers, class_count, dtype)
    elif arch == "cnn":
        model = CNN(class_count, tau, **llkw)
    elif arch == "gcn":
        model = GCN(num_node_features=in_dim, num_classes=class_count)
    elif arch == "gcn_cora_baseline":
        model = GCNCoraBaseline(num_node_features=in_dim, num_classes=class_count).to(
            device
        )
    elif arch == "gat_cora_baseline":
        model = GATBaseline(
            in_channels=in_dim, hidden_channels=16, out_channels=class_count
        ).to(device)
    elif arch == "gat":
        model = GATDifflog(
            in_channels=in_dim, hidden_channels=16, out_channels=class_count
        ).to(device)
    else:
        raise NotImplementedError(arch)

    model = model.to(llkw["device"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer
