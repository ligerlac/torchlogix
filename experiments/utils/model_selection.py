import torch
import torchlogix
from hist import Hist
import numpy as np


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
            "residual_probability": args.residual_probability,
            },
        "device": args.device,
        "lut_rank": args.lut_rank,
        "thresholds": thresholds,
        "binarization": args.binarization,
        "binarization_kwargs": {
            "one_per": args.binarization_per,
            "temperature_sampling": args.binarization_temperature,
            "temperature_softplus": args.binarization_temperature_softplus,
            "forward_sampling": args.binarization_forward_sampling
            }
    }
    if args.parametrization == "dwn":
        try:
            import utils.dwn_models
            model_cls = utils.dwn_models.__dict__[args.architecture]
        except ImportError:
            raise ImportError("DWN models require the 'dwn' package. Please install it to use these models.")
    else:
        model_cls = torchlogix.models.__dict__[args.architecture]
    model = model_cls(**llkw)

    model = model.to(llkw["device"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")
    print(model)

    # for layer in model:
    #     if not hasattr(layer, "get_luts_and_ids"):
    #         continue
    #     lut_ids = layer.get_luts_and_ids()[1]
    #     # put into single array if it's a list of arrays
    #     if isinstance(lut_ids, list):
    #         lut_ids = np.concatenate([np.concatenate([l.cpu().numpy() for l in lut_ids_], axis=0) for lut_ids_ in lut_ids], axis=0)
    #     else:
    #         lut_ids = lut_ids.cpu().numpy()
    #     h = Hist.new.Regular(16, 0, 16, name="x").Double()
    #     h.fill(lut_ids)
    #     print(h)

    loss_fn = torch.nn.CrossEntropyLoss()

    if args.binarization_learning_rate and isinstance(model[0], torchlogix.layers.LearnableBinarization):
        binarization_params = list(model[0].parameters())
        other_params = [p for p in model.parameters() if p not in set(binarization_params)]
        
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': args.learning_rate},
            {'params': binarization_params, 'lr': args.binarization_learning_rate * args.learning_rate}
        ])
    else:
        if args.binarization_learning_rate:
            print("Warning: binarization_learning_rate specified but the model does not use LearnableBinarization. Ignoring this parameter.")
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optim
