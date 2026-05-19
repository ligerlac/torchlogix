"""Tests for export mode functionality (ONNX/TorchScript tracing)."""
import tempfile
import os
import warnings
import operator

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from torchlogix.layers import (
    GroupSum,
    LogicConv2d,
    LogicConv3d,
    LogicDense,
    OrPooling2d,
    OrPooling3d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logic_dense_model():
    model = nn.Sequential(
        LogicDense(16, 32, parametrization="raw"),
        LogicDense(32, 16, parametrization="raw"),
    )
    model.eval()
    return model


@pytest.fixture
def conv2d_model_wo_group_sum():
    model = nn.Sequential(
        # LogicConv2d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2),
        LogicConv2d(in_dim=8, channels=3, num_kernels=7, receptive_field_size=3, tree_depth=2),
        OrPooling2d(kernel_size=2, stride=2),
        nn.Flatten(),          # 3 × 3 × 8 = 72
        # LogicDense(72, 64, parametrization="raw"),
        LogicDense(63, 64, parametrization="raw"),
        LogicDense(64, 50, parametrization="raw"),
    )
    model.eval()
    return model


@pytest.fixture
def conv3d_model_wo_group_sum():
    model = nn.Sequential(
        LogicConv3d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2),
        OrPooling3d(kernel_size=2, stride=2),
        nn.Flatten(),          # 3 × 3 × 3 × 8 = 216
        LogicDense(216, 128, parametrization="raw"),
        LogicDense(128, 64, parametrization="raw"),
    )
    model.eval()
    return model

@pytest.fixture
def conv2d_model():
    model = nn.Sequential(
        LogicConv2d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2),
        OrPooling2d(kernel_size=2, stride=2),
        nn.Flatten(),          # 3 × 3 × 8 = 72
        LogicDense(72, 64, parametrization="raw"),
        LogicDense(64, 50, parametrization="raw"),
        GroupSum(10),
    )
    model.eval()
    return model


@pytest.fixture
def conv3d_model():
    model = nn.Sequential(
        LogicConv3d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2),
        OrPooling3d(kernel_size=2, stride=2),
        nn.Flatten(),          # 3 × 3 × 3 × 8 = 216
        LogicDense(216, 128, parametrization="raw"),
        LogicDense(128, 64, parametrization="raw"),
        GroupSum(8),
    )
    model.eval()
    return model


@pytest.fixture
def sample_input_1d():
    torch.manual_seed(0)
    return torch.randint(0, 2, (8, 16)).bool()


@pytest.fixture
def sample_input_2d():
    torch.manual_seed(0)
    return torch.randint(0, 2, (8, 3, 8, 8)).bool()


@pytest.fixture
def sample_input_3d():
    torch.manual_seed(0)
    return torch.randint(0, 2, (4, 3, 8, 8, 8)).bool()


def _enable_export_mode(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "set_export_mode"):
            module.set_export_mode(True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onnx_run(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Export *model* to ONNX (temp file) and run inference via OnnxRuntime."""
    batch = torch.export.Dim("batch")
    dynamic_shapes = {"input": {0: batch}}

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
        print(f"Exporting model to ONNX at {tmp_path}...")

    try:
        torch.onnx.export(
            model,
            (x,),                          # args must be a tuple with dynamic_shapes
            tmp_path,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=dynamic_shapes,
        )
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        (output_name,) = [o.name for o in sess.get_outputs()]
        return sess.run([output_name], {"input": x.numpy()})[0]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Parametrize over both 2-D and 3-D fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_fixture, input_fixture",
    [
        ("logic_dense_model", "sample_input_1d"),
        ("conv2d_model_wo_group_sum", "sample_input_2d"),
        # ("conv3d_model_wo_group_sum", "sample_input_3d"),
        # ("conv2d_model", "sample_input_2d"),
        # ("conv3d_model", "sample_input_3d"),
    ],
)
class TestExportModeEquivalence:
    """Eval-mode and export-mode must agree on binary inputs."""

    def test_eval_export_equivalence(self, model_fixture, input_fixture, request):
        model = request.getfixturevalue(model_fixture)
        x = request.getfixturevalue(input_fixture)

        # Baseline: plain eval-mode forward (accepts float bool-valued tensors)
        x_float = x.float()
        result_eval = model(x_float)

        # Export mode
        _enable_export_mode(model)
        result_export = model(x)

        assert torch.allclose(result_eval, result_export.float(), atol=1e-6), (
            f"[{model_fixture}] eval and export results diverge"
        )


ALLOWED_ONNX_OPS_WO_GROUP_SUM = {
    # Boolean logic
    "And", "Or", "Not", "Xor",
    # Comparisons / selection (from lut_ids == n masks)
    "Equal",
    # Structural / indexing
    "Gather", "GatherElements", "GatherND",
    "Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze",
    "Slice", "Concat", "Shape", "Expand", "Tile", "Pad",
    "Identity", "Constant", "ConstantOfShape",
}

ALLOWED_ONNX_OPS = ALLOWED_ONNX_OPS_WO_GROUP_SUM | {
    "Cast", "ReduceSum", # for group sum
}


def _onnx_ops(tmp_path: str) -> set[str]:
    """Return the set of op types present in an ONNX model file."""
    import onnx
    model_proto = onnx.load(tmp_path)
    return {node.op_type for node in model_proto.graph.node}


class TestOnnxExport:
    """ONNX round-trip and op-set purity."""

    @pytest.mark.parametrize("model_fixture, input_fixture", [
        ("logic_dense_model", "sample_input_1d"),
        ("conv2d_model_wo_group_sum", "sample_input_2d"),
        ("conv3d_model_wo_group_sum", "sample_input_3d"),
        ("conv2d_model", "sample_input_2d"),
        ("conv3d_model", "sample_input_3d"),
    ])
    def test_onnx_roundtrip(self, model_fixture, input_fixture, request):
        model = request.getfixturevalue(model_fixture)
        x = request.getfixturevalue(input_fixture)
        _enable_export_mode(model)

        eager_out = model(x).float().detach().numpy()
        onnx_out = _onnx_run(model, x)

        assert np.allclose(eager_out, onnx_out, atol=1e-5)

    @pytest.mark.parametrize("model_fixture, input_fixture, allowed_ops", [
        ("logic_dense_model", "sample_input_1d", ALLOWED_ONNX_OPS_WO_GROUP_SUM),
        ("conv2d_model_wo_group_sum", "sample_input_2d", ALLOWED_ONNX_OPS_WO_GROUP_SUM),
        ("conv3d_model_wo_group_sum", "sample_input_3d", ALLOWED_ONNX_OPS_WO_GROUP_SUM),
        ("conv2d_model", "sample_input_2d", ALLOWED_ONNX_OPS),
        ("conv3d_model", "sample_input_3d", ALLOWED_ONNX_OPS),
    ])
    def test_onnx_ops_are_pure_logic(self, model_fixture, input_fixture, allowed_ops, request, tmp_path):
        model = request.getfixturevalue(model_fixture)
        x = request.getfixturevalue(input_fixture)
        _enable_export_mode(model)

        onnx_path = str(tmp_path / "model.onnx")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
            torch.onnx.export(
                model, (x,), onnx_path,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_shapes={"input": {0: torch.export.Dim("batch")}},
            )

        ops = _onnx_ops(onnx_path)
        disallowed = ops - allowed_ops
        assert not disallowed, (
            f"ONNX graph contains non-logic ops: {disallowed}\n"
            f"Full op set: {ops}"
        )


ALLOWED_FX_TARGETS = {
    # Logic ops
    torch.ops.aten.__and__.Tensor,
    torch.ops.aten.__or__.Tensor,
    torch.ops.aten.__xor__.Tensor,
    torch.ops.aten.bitwise_not.default,
    # LUT dispatch
    torch.ops.aten.where.self,
    torch.ops.aten.eq.Scalar,
    # Indexing / wiring
    torch.ops.aten.index.Tensor,
    torch.ops.aten.select.int,
    # Shape bookkeeping (final reshape only)
    torch.ops.aten.reshape.default,
    # Constant construction (LUT 0 and 15)
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.ones_like.default,
}


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


class TestFXGraphPurity:

    @pytest.mark.parametrize("model_fixture, input_fixture", [
        ("logic_dense_model", "sample_input_1d"),
        ("conv2d_model_wo_group_sum", "sample_input_2d"),
        ("conv3d_model_wo_group_sum", "sample_input_3d"),
        # ("conv2d_model", "sample_input_2d"),
        # ("conv3d_model", "sample_input_3d"),
    ])
    def test_fx_graph_is_pure_logic(self, model_fixture, input_fixture, request):
        model = request.getfixturevalue(model_fixture)
        x = request.getfixturevalue(input_fixture)
        _enable_export_mode(model)

        exported = torch.export.export(model, (x,), strict=False)
        gm = exported.module()
        gm = constant_fold_views(gm)

        disallowed = []
        for node in gm.graph.nodes:
            if node.op == 'call_function' and node.target not in ALLOWED_FX_TARGETS:
                disallowed.append(f"{node.name}: {node.target}")

        assert not disallowed, (
            f"[{model_fixture}] FX graph contains non-logic ops:\n"
            + "\n".join(disallowed)
        )

# @pytest.mark.parametrize(
#     "model_fixture, input_fixture",
#     [
#         ("logic_dense_model", "sample_input_1d"),
#         ("conv2d_model_wo_group_sum", "sample_input_2d"),
#         ("conv3d_model_wo_group_sum", "sample_input_3d"),
#         ("conv2d_model", "sample_input_2d"),
#         ("conv3d_model", "sample_input_3d"),
#     ],
# )
# class TestExportModeEquivalenceNumpy:
#     """Numpy and torch export-mode must agree on identical inputs."""

#     def test_numpy_export_equivalence(self, model_fixture, input_fixture, request):
#         model = request.getfixturevalue(model_fixture)
#         x = request.getfixturevalue(input_fixture)
#         _enable_export_mode(model)

#         x_np = x.numpy()
#         result_torch = model(x)
#         result_numpy = model(x_np)

#         assert isinstance(result_numpy, np.ndarray), "Numpy input should produce numpy output"
#         assert np.allclose(result_torch.numpy(), result_numpy, atol=1e-6), (
#             f"[{model_fixture}] torch and numpy export results diverge"
#         )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
