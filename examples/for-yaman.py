import torch
import torch.nn as nn
from torchlogix.layers import GroupSum, LogicConv2d, LogicConv3d, LogicDense, OrPooling2d
from torchlogix.circuit import Circuit
from torchlogix.utils import set_export_mode
from datetime import datetime


# w/ custom forward pass. previously not possible to export
class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=32, channels=3, num_kernels=8,
                    receptive_field_size=3, tree_depth=2,
                    parametrization_kwargs={"weight_init": "random"}) # 8 x 30 x 30 = 7200
        self.pool = OrPooling2d(kernel_size=2, stride=2) # 8 x 15 x 15 = 1800
        self.dense = LogicDense(1801, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"})
        self.group_sum = GroupSum(10)
        self.input_shape = (32*32*3 + 1,)

    def forward(self, x):
        assert x.shape[1:] == (32*32*3 + 1,)
        img, feat = x[:, :-1].reshape(-1, 3, 32, 32), x[:, -1:]
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x
    

# from the paper "Convolutional Differentiable Logic Gate Networks"
class ClgnCifar(nn.Sequential):
    k = None
    n_bits = None
    tau = None
    llkw = {"parametrization_kwargs": {"weight_init": "random"}}
    def __init__(self):
        self.input_shape = (3*self.n_bits, 32, 32)
        super().__init__(
            LogicConv2d(
                in_dim=32,
                num_kernels=self.k,
                channels=3*self.n_bits,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2), # kx16x
            LogicConv2d(
                in_dim=16,
                channels=self.k,
                num_kernels=4*self.k,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2),
            LogicConv2d(
                in_dim=8,
                channels=4*self.k,
                num_kernels=16*self.k,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2),
            LogicConv2d(
                in_dim=4,
                channels=16*self.k,
                num_kernels=32*self.k,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            LogicDense(in_dim=128*self.k, out_dim=1280*self.k, **self.llkw),
            LogicDense(in_dim=1280*self.k, out_dim=640*self.k, **self.llkw),
            LogicDense(in_dim=640*self.k, out_dim=320*self.k, **self.llkw),
            GroupSum(k=10, tau=self.tau),
        )


class ClgnCifarSmall(ClgnCifar):
    k = 16
    n_bits = 2
    tau = 20


class ClgnCifarMedium(ClgnCifar):
    k = 256
    n_bits = 2
    tau = 20


if __name__ == "__main__":

    model = ClgnCifarSmall()
    # model = BranchModel()

    BATCH_SIZE = 160_000
    x = torch.randint(0, 2, (BATCH_SIZE, *model.input_shape), dtype=torch.bool)
    x_np = x.cpu().numpy()

    model.eval()
    preds_model_eval = model(x[:1])

    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=model.input_shape)

    print(f"The original circuit has {len(circuit.gates)} gates")
    t0 = datetime.now()
    circuit.compile()
    print(f"Compiling the original circuit took {(datetime.now() - t0).total_seconds():.4f} seconds")

    t0 = datetime.now()
    preds_original_circuit = circuit(x_np, use_compiled=True)
    print(f"Compiled circuit inference time: {(datetime.now() - t0).total_seconds():.4f} seconds")

    t0 = datetime.now()
    circuit.simplify()
    print(f"Simplifying the circuit took {(datetime.now() - t0).total_seconds():.4f} seconds")
    print(f"The simplified circuit has {len(circuit.gates)} gates")

    t0 = datetime.now()
    circuit.compile()
    print(f"Compiling the simplified circuit took {(datetime.now() - t0).total_seconds():.4f} seconds")

    t0 = datetime.now()
    preds_simplified_circuit = circuit(x_np, use_compiled=True)
    print(f"Compiled simplified circuit inference time: {(datetime.now() - t0).total_seconds():.4f} seconds")

    # assert that preds_model_eval, preds_original_circuit, and preds_simplified_circuit are identical
    assert torch.allclose(preds_model_eval, torch.from_numpy(preds_original_circuit[:1])), "Outputs differ between model and original circuit!"
    assert torch.allclose(preds_model_eval, torch.from_numpy(preds_simplified_circuit[:1])), "Outputs differ between model and simplified circuit!"
