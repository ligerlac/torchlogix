import sys
import torch
import torch.nn as nn
from torchlogix.layers import GroupSum, LogicConv2d, LogicConv3d, LogicDense, OrPooling2d


import torch
from torch.fx import Interpreter
import operator


def constant_fold_views(gm: torch.fx.GraphModule):
    env = {}

    def get_attr_value(gm, target: str):
        obj = gm
        for attr in target.split('.'):
            obj = getattr(obj, attr)
        return obj

    VIEW_OPS = {
        torch.ops.aten.movedim.int,
        torch.ops.aten.reshape.default,
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.moveaxis.int,
        torch.ops.aten.unbind.int,
        torch.ops.aten.lift_fresh_copy.default,
    }

    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            continue
        if node.op == 'get_attr':
            env[node] = get_attr_value(gm, node.target)
            continue
        if node.op == 'call_function' and node.target in VIEW_OPS:
            args_resolved = []
            all_const = True
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    if a in env:
                        args_resolved.append(env[a])
                    else:
                        all_const = False
                        break
                else:
                    args_resolved.append(a)

            if all_const:
                result = node.target(*args_resolved, **node.kwargs)
                env[node] = result

    # Replace folded nodes with constants
    for node, value in env.items():
        if node.op in ('placeholder', 'get_attr'):
            continue

        const_name = f"_folded_{node.name}"

        if isinstance(value, torch.Tensor):
            gm.register_buffer(const_name, value)
            with gm.graph.inserting_before(node):
                new_node = gm.graph.get_attr(const_name)
            node.replace_all_uses_with(new_node)

        elif isinstance(value, (tuple, list)):
            # e.g. unbind returns a tuple of tensors
            # replace getitem users directly
            for user in list(node.users):
                if user.op == 'call_function' and user.target is operator.getitem:
                    idx = user.args[1]
                    item = value[idx]
                    item_name = f"_folded_{node.name}_{idx}"
                    if isinstance(item, torch.Tensor):
                        gm.register_buffer(item_name, item)
                        with gm.graph.inserting_before(user):
                            new_node = gm.graph.get_attr(item_name)
                        user.replace_all_uses_with(new_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=32, channels=3, num_kernels=8,
                    receptive_field_size=3, tree_depth=2,
                    parametrization_kwargs={"weight_init": "random"}) # 8 x 30 x 30 = 7200
        self.pool = OrPooling2d(kernel_size=2, stride=2) # 8 x 15 x 15 = 1800
        self.dense = LogicDense(1801, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"})
        self.group_sum = GroupSum(10)

    def forward(self, x):
        assert x.shape[1:] == (32*32*3 + 1,)
        img, feat = x[:, :-1].reshape(-1, 3, 32, 32), x[:, -1:]
        print(f"img shape: {img.shape}, feat shape: {feat.shape}")
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x


if __name__ == "__main__":

    # Build the same model as in the original script
    from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d

    # model = nn.Sequential(
    #     LogicConv2d(in_dim=8, channels=3, num_kernels=8,
    #                 receptive_field_size=3, tree_depth=2,
    #                 parametrization_kwargs={"weight_init": "random"}),
    #     OrPooling2d(kernel_size=2, stride=2),
    #     nn.Flatten(),
    #     LogicDense(72, 64, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
    #     LogicDense(64, 50, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
    #     GroupSum(10),
    # )

    model = MyModel()

    model.eval()
    for module in model.modules():
        if hasattr(module, "set_export_mode"):
            module.set_export_mode(True)

    torch.manual_seed(42)
    # x = torch.randint(0, 2, (1, 8, 8, 8), dtype=torch.bool)
    x = torch.randint(0, 2, (1, 32*32*3 + 1), dtype=torch.bool)

    exported = torch.export.export(model, (x,), strict=False)
    gm = exported.module()
    gm = constant_fold_views(gm)
    # for node in gm.graph.nodes:
    #     print(f"{node.name}: {node.op} {node.target} {node.args} {node.kwargs}")

    from circuit_ir import serialize_circuit

    circuit = serialize_circuit(
        gm,
        input_shape=list(x.shape),
        lisp_path="circuit.ir",
        c_path="circuit.c",
    )        

    # exported = torch.export.export(model, (x,), strict=False)
    # state_dict = dict(exported.state_dict)

    # # print(f"{state_dict=}")
    # for node in exported.graph.nodes:
    #     print(f"{node.name}: {node.op} {node.target} {node.args} {node.kwargs}")

    # -----------------------------------------------------------------------
    # Compile circuit.c and verify functional equivalence against the model
    # -----------------------------------------------------------------------
    import subprocess
    import ctypes

    print("\n--- Compiling circuit.c ---")
    compile_result = subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", "circuit.so", "circuit.c"],
        capture_output=True, text=True,
    )
    if compile_result.returncode != 0:
        print("Compilation FAILED:")
        print(compile_result.stderr)
        sys.exit(1)
    print("Compilation successful.")

    lib = ctypes.CDLL("./circuit.so")
    n_inputs  = circuit.n_inputs
    n_outputs = len(circuit.output_ids)
    lib.circuit_eval.argtypes = [
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_bool),
    ]
    lib.circuit_eval.restype = None

    N_TRIALS   = 200
    mismatches = 0
    print(f"\n--- Equivalence check: {N_TRIALS} random inputs "
          f"({n_inputs} in → {n_outputs} out) ---")

    # Determine if the model applies a post-processing reduction (e.g. GroupSum)
    # by comparing the number of model outputs to the number of circuit outputs.
    with torch.no_grad():
        _sample_out = gm(x).flatten()
    n_model_out = _sample_out.numel()
    uses_group_sum = (n_model_out < n_outputs)
    if uses_group_sum:
        group_size = n_outputs // n_model_out
        print(f"  (GroupSum detected: {n_outputs} circuit outputs → "
              f"{n_model_out} model outputs, group size {group_size})")

    torch.manual_seed(0)
    for trial in range(N_TRIALS):
        x_rand = torch.randint(0, 2, x.shape, dtype=torch.bool)

        with torch.no_grad():
            y_torch_flat = gm(x_rand).flatten().tolist()

        in_arr  = (ctypes.c_bool * n_inputs)(*x_rand.flatten().tolist())
        out_arr = (ctypes.c_bool * n_outputs)()
        lib.circuit_eval(in_arr, out_arr)
        y_c = [bool(out_arr[k]) for k in range(n_outputs)]

        if uses_group_sum:
            # Sum groups of booleans to match model's GroupSum output
            y_c_cmp   = [sum(y_c[k * group_size:(k + 1) * group_size])
                         for k in range(n_model_out)]
            y_ref_cmp = [int(round(v)) for v in y_torch_flat]
        else:
            y_c_cmp   = y_c
            y_ref_cmp = [bool(v) for v in y_torch_flat]

        if y_c_cmp != y_ref_cmp:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Trial {trial}: MISMATCH")
                print(f"    PyTorch: {y_ref_cmp[:10]}...")
                print(f"    C:       {y_c_cmp[:10]}...")

    if mismatches == 0:
        print(f"PASSED — all {N_TRIALS} trials matched. Circuit is functionally equivalent.")
    else:
        print(f"FAILED — {mismatches}/{N_TRIALS} trials had mismatches.")
        sys.exit(1)
