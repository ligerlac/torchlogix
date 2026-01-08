import torch
import torchlogix


def get_model(thresholds, args):
    """
    Select model from the architecture.
    It can be a difflogic model or a baseline model.
    """
    llkw = {
        "connections": args.connections,
        "connections_kwargs": {
            "init_method": args.connections_init_method,
            "temperature": args.connections_temperature,
            "num_candidates": args.connections_num_candidates,
            "gumbel": args.connections_gumbel
            },
        "parametrization": args.parametrization,
        "parametrization_kwargs": {
            "temperature": args.parametrization_temperature,
            "forward_sampling": args.forward_sampling,
            "weight_init": args.weight_init,
            "residual_param": args.residual_param,
            "arbitrary_basis": args.arbitrary_basis
            },
        "device": args.device,
        "lut_rank": args.lut_rank,
        "thresholds": thresholds,
        "binarization": args.binarization,
        "binarization_kwargs": {
            "feature_wise": args.binarization_feature_wise,
            "temperature_sampling": args.binarization_temperature,
            "temperature_softplus": args.binarization_temperature_softplus,
            "forward_sampling": args.forward_sampling
            }
    }
    model_cls = torchlogix.models.__dict__[args.architecture]
    model = model_cls(**llkw)

    model = model.to(llkw["device"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer
