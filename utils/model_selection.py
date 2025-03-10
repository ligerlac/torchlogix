import torch
from utils.loading import input_dim_of_dataset, num_classes_of_dataset
from models.base_linear import BaseModel
from models.conv import CNN
from models.gcn_conv import GCN, GCNCoraBaseline

IMPL_TO_DEVICE = {
    'cuda': 'cuda',
    'python': 'cpu'
}

def get_model(args):
    llkw = {
        'grad_factor': args.grad_factor,
        'connections': args.connections,
        'implementation': args.implementation,
        'device': IMPL_TO_DEVICE[args.implementation],

    }
    device = IMPL_TO_DEVICE[args.implementation]
    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    arch = args.architecture
    k = args.num_neurons
    l = args.num_layers
    if arch == 'randomly_connected':
        model = BaseModel(in_dim, k, l, class_count, args, **llkw)
    elif arch == 'cnn':
        model = CNN(class_count, args, **llkw)
    elif arch == 'gcn':
        model = GCN(num_node_features=1433, num_classes=7).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}")
    elif arch == 'gcn_cora_baseline':
        model = GCNCoraBaseline(num_node_features=1433, num_classes=7).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}")
    else:
        raise NotImplementedError(arch)

    model = model.to(llkw['device'])    # not implemented for cnn
    # total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    # print(f'total_num_neurons={total_num_neurons}')
    # total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    # print(f'total_num_weights={total_num_weights}')
    # if args.experiment_id is not None:
    #     results.store_results({
    #         'total_num_neurons': total_num_neurons,
    #         'total_num_weights': total_num_weights,
    #     })

    print(model)
    # for name, param in model.named_parameters():
    #     if "tree_weights" in name:
    #         print(
    #             f"{name}: Mean {param.mean().item()}, Grad mean"
    #             f" {param.grad.mean().item() if param.grad is not None else 'None'}")
    # if args.experiment_id is not None:
    #     results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer