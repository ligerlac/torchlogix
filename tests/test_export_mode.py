"""Tests for export mode functionality (ONNX/TorchScript tracing)."""
import tempfile
import os
import warnings
import operator

import numpy as np
import pytest
import torch
import torch.nn as nn

import onnx
from onnx import shape_inference

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
        LogicDense(16, 32, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
        LogicDense(32, 16, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
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
def single_3d_conv_model():
    model = nn.Sequential(
        LogicConv3d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2),
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


def _enable_export_mode(model: nn.Module, batch_size: int = 1) -> None:
    for module in model.modules():
        if hasattr(module, "set_export_mode"):
            module.set_export_mode(True, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _onnx_run(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    import onnxruntime as ort
    """Export *model* to ONNX (temp file) and run inference via OnnxRuntime."""
    # batch = torch.export.Dim("batch")
    # dynamic_shapes = {"input": {0: batch}}

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
        print(f"Exporting model to ONNX at {tmp_path}...")

    try:
        torch.onnx.export(
            model,
            (x,),                          # args must be a tuple with dynamic_shapes
            tmp_path,
            opset_version=21,
            input_names=["input"],
            output_names=["output"],
            verbose=True,
            # dynamic_shapes=dynamic_shapes,
        )
        
        model = onnx.load(tmp_path)
        onnx.checker.check_model(model)
        inferred = shape_inference.infer_shapes(model)

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
        ("conv3d_model_wo_group_sum", "sample_input_3d"),
        ("conv2d_model", "sample_input_2d"),
        ("conv3d_model", "sample_input_3d"),
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
        _enable_export_mode(model, batch_size=x.shape[0])
        result_export = model(x)

        assert torch.allclose(result_eval, result_export.float(), atol=1e-6), (
            f"[{model_fixture}] eval and export results diverge"
        )


ALLOWED_ONNX_OPS = {
    # Boolean logic
    "And", "Or", "Not", "Xor",
    # Comparisons / selection (from lut_ids == n masks)
    "Equal",
    # Structural / indexing
    "Gather", "GatherElements", "GatherND",
    "Where", "NonZero",
    "Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze",
    "Slice", "Concat", "Shape", "Expand", "Tile", "Pad",
    "Identity", "Constant", "ConstantOfShape",
}

ALLOWED_ONNX_OPS_GROUP_SUM = {
    "Cast", "ReduceSum", # for group sum
}


def _onnx_ops(tmp_path: str) -> set[str]:
    """Return the set of op types present in an ONNX model file."""
    import onnx
    model_proto = onnx.load(tmp_path)
    return {node.op_type for node in model_proto.graph.node}


class TestOnnxExport:
    """ONNX round-trip and op-set purity."""

    # Round-trip doesnt work for now because ONNX runtime
    # doesn't support advanced indexing (bare ONNX does)

    """
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
        _enable_export_mode(model, batch_size=x.shape[0])

        eager_out = model(x).float().detach().numpy()
        onnx_out = _onnx_run(model, x)

        assert np.allclose(eager_out, onnx_out, atol=1e-5)
    """

    @pytest.mark.parametrize("model_fixture, input_fixture, allowed_ops", [
        ("logic_dense_model", "sample_input_1d", ALLOWED_ONNX_OPS),
        ("conv2d_model_wo_group_sum", "sample_input_2d", ALLOWED_ONNX_OPS),
        ("conv3d_model_wo_group_sum", "sample_input_3d", ALLOWED_ONNX_OPS),
        ("conv2d_model", "sample_input_2d", ALLOWED_ONNX_OPS | ALLOWED_ONNX_OPS_GROUP_SUM),
        ("conv3d_model", "sample_input_3d", ALLOWED_ONNX_OPS | ALLOWED_ONNX_OPS_GROUP_SUM),
        ("single_3d_conv_model", "sample_input_3d", ALLOWED_ONNX_OPS),
    ])
    def test_onnx_ops_are_pure_logic(self, model_fixture, input_fixture, allowed_ops, request, tmp_path):
        model = request.getfixturevalue(model_fixture)
        x = request.getfixturevalue(input_fixture)
        _enable_export_mode(model, batch_size=x.shape[0])

        onnx_path = str(tmp_path / "model.onnx")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
            torch.onnx.export(
                model, (x,), onnx_path,
                opset_version=21,
                input_names=["input"],
                output_names=["output"],
                # dynamic_shapes={"input": {0: torch.export.Dim("batch")}},
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

    # Custom torchlogix ops (registered via torch.library.custom_op)
    torch.ops.torchlogix.lut_layer.default,

    # LUT ops (kept for backward compat; not emitted with custom ops)
    torch.ops.aten.where.self,
    torch.ops.aten.eq.Scalar,

    # Comparisons (needed for export guards)
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.le.Scalar,
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.lt.Scalar,
    operator.ge,
    operator.le,
    operator.gt,
    operator.lt,
    operator.getitem,

    # Indexing / wiring
    torch.ops.aten.index.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.unbind.int,

    # Shape / layout (view ops)
    torch.ops.aten.reshape.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.moveaxis.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.pad.default,
    torch.ops.aten.unfold.default,

    # Advanced view variants
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.stack.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.chunk.default,

    # Index writes
    torch.ops.aten.index_put_.default,

    # Constants and copies
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.ones_like.default,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.lift_fresh_copy.default,

    # Symbolic shape system (export internals)
    torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_constrain_range_for_size.default,
    torch.ops.aten._assert_scalar.default,
}

ALLOWED_FX_TARGETS_GROUP_SUM = {
    # Custom torchlogix group_sum op
    torch.ops.torchlogix.group_sum.default,
    # Fallback aten ops (kept for backward compat)
    torch.ops.aten.add.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.sum.dim_IntList,
}


class TestFXGraphPurity:

    @pytest.mark.parametrize("model_fixture, input_fixture, allowed_targets", [
        ("logic_dense_model", "sample_input_1d", ALLOWED_FX_TARGETS),
        ("conv2d_model_wo_group_sum", "sample_input_2d", ALLOWED_FX_TARGETS),
        ("conv3d_model_wo_group_sum", "sample_input_3d", ALLOWED_FX_TARGETS),
        ("conv2d_model", "sample_input_2d", ALLOWED_FX_TARGETS | ALLOWED_FX_TARGETS_GROUP_SUM),
        ("conv3d_model", "sample_input_3d", ALLOWED_FX_TARGETS | ALLOWED_FX_TARGETS_GROUP_SUM),
    ])
    def test_fx_graph_is_pure_logic(self, model_fixture, input_fixture, allowed_targets, request):
        model = request.getfixturevalue(model_fixture)
        x = request.getfixturevalue(input_fixture)
        _enable_export_mode(model, batch_size=x.shape[0])

        exported = torch.export.export(model, (x,), strict=False)
        gm = exported.module()

        disallowed = []
        for node in gm.graph.nodes:
            if node.op == 'call_function' and node.target not in allowed_targets:
                disallowed.append(f"{node.name}: {node.target}")

        assert not disallowed, (
            f"[{model_fixture}] FX graph contains non-logic ops:\n"
            + "\n".join(disallowed)
        )


@pytest.mark.parametrize(
    "layer, input_shape",
    [
        (LogicDense(128, 128, parametrization_kwargs={"weight_init": "random"}), (8, 128)),
        (LogicConv2d(in_dim=(12, 8), channels=3, num_kernels=8, receptive_field_size=(3, 2), tree_depth=2, parametrization_kwargs={"weight_init": "random"}), (7, 3, 12, 8)),
        (LogicConv3d(in_dim=(16, 14, 12), channels=3, num_kernels=8, receptive_field_size=(4, 3, 2), tree_depth=2, parametrization_kwargs={"weight_init": "random"}), (11, 3, 16, 14, 12)),
        (OrPooling2d(kernel_size=2, stride=2), (8, 3, 8, 8)),
        (OrPooling3d(kernel_size=2, stride=2), (8, 3, 8, 8, 8)),
        (GroupSum(10), (8, 50)),
    ],
)
def test_numpy_export_equivalence(layer, input_shape):
    """Numpy and torch export-mode must agree on identical inputs."""
    x = torch.randint(0, 2, input_shape).bool()
    _enable_export_mode(layer, batch_size=x.shape[0])

    x_np = x.numpy()
    result_torch = layer(x)
    result_numpy = layer(x_np)

    assert isinstance(result_numpy, np.ndarray), "Numpy input should produce numpy output"
    assert np.allclose(result_torch.numpy(), result_numpy, atol=1e-6), (
        f"[{layer}] torch and numpy export results diverge"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
