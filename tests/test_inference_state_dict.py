import pytest
import torch

from torchlogix import get_inference_state_dict, load_inference_state_dict
from torchlogix.layers import FixedBinarization, GroupSum, LogicConv2d, LogicDense
from torchlogix.models.nn import RandomlyConnectedNN


@torch.no_grad()
def test_logic_dense_inference_only_round_trip():
    layer = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )
    layer.eval()

    x_bool = torch.randint(0, 2, (16, 8), dtype=torch.bool)
    expected = layer(x_bool.float())

    inference_state = layer.state_dict(inference_only=True)
    assert "_inference.lut_ids" in inference_state
    assert "weight" not in inference_state

    restored = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )
    restored.load_state_dict(inference_state, inference_only=True)

    actual = restored(x_bool)
    assert restored.export_mode is True
    assert restored.inference_only is True
    assert torch.allclose(expected, actual.float(), atol=1e-6)


@torch.no_grad()
def test_logic_dense_inference_only_round_trip_with_learnable_connections():
    layer = LogicDense(
        in_dim=8,
        out_dim=6,
        connections="learnable",
        connections_kwargs={"num_candidates": 2},
    )
    layer.eval()

    x_bool = torch.randint(0, 2, (20, 8), dtype=torch.bool)
    expected = layer(x_bool.float())

    inference_state = layer.state_dict(inference_only=True)
    assert "_inference.connection_candidates" in inference_state
    assert "_inference.connection_selection" in inference_state

    restored = LogicDense(
        in_dim=8,
        out_dim=6,
        connections="learnable",
        connections_kwargs={"num_candidates": 2},
    )
    restored.load_state_dict(inference_state, inference_only=True)

    actual = restored(x_bool)
    assert torch.allclose(expected, actual.float(), atol=1e-6)


@torch.no_grad()
def test_logic_conv_inference_only_round_trip():
    layer = LogicConv2d(
        in_dim=6,
        channels=1,
        num_kernels=3,
        tree_depth=2,
        receptive_field_size=3,
        padding=1,
        parametrization="raw",
        connections_kwargs={"init_method": "random-unique"},
    )
    layer.eval()

    x_bool = torch.randint(0, 2, (10, 1, 6, 6), dtype=torch.bool)
    expected = layer(x_bool.float())

    inference_state = layer.state_dict(inference_only=True)
    assert "_inference.tree_lut_ids.0" in inference_state
    assert "_inference.connection_indices.0" in inference_state

    restored = LogicConv2d(
        in_dim=6,
        channels=1,
        num_kernels=3,
        tree_depth=2,
        receptive_field_size=3,
        padding=1,
        parametrization="raw",
        connections_kwargs={"init_method": "random-unique"},
    )
    restored.load_state_dict(inference_state, inference_only=True)

    actual = restored(x_bool)
    assert restored.export_mode is True
    assert restored.inference_only is True
    assert torch.equal(expected.bool(), actual)


@torch.no_grad()
def test_model_inference_only_round_trip():
    model = RandomlyConnectedNN(
        in_dim=8,
        k=8,
        layers=2,
        class_count=2,
        tau=1.0,
        device="cpu",
        connections="fixed",
        connections_kwargs={"init_method": "random-unique"},
    )
    model.eval()

    x_bool = torch.randint(0, 2, (12, 8), dtype=torch.bool)
    expected = model(x_bool.float())

    inference_state = model.state_dict(inference_only=True)
    restored = RandomlyConnectedNN(
        in_dim=8,
        k=8,
        layers=2,
        class_count=2,
        tau=1.0,
        device="cpu",
        connections="fixed",
        connections_kwargs={"init_method": "random-unique"},
    )
    restored.load_state_dict(inference_state, inference_only=True)

    actual = restored(x_bool)
    assert torch.allclose(expected, actual.float(), atol=1e-6)


@torch.no_grad()
def test_inference_only_state_preserves_thresholding_via_generic_helpers():
    thresholds = torch.tensor(
        [
            [0.2, 0.7],
            [0.4, 0.8],
        ],
        dtype=torch.float32,
    )
    model = torch.nn.Sequential(
        FixedBinarization(thresholds=thresholds),
        torch.nn.Flatten(),
        LogicDense(
            in_dim=4,
            out_dim=6,
            parametrization="light",
            connections_kwargs={"init_method": "random-unique"},
        ),
        GroupSum(2),
    )
    model.eval()

    x = torch.rand(10, 2)
    expected = model(x)

    inference_state = get_inference_state_dict(model)
    assert "0._inference.thresholds" in inference_state
    assert "2._inference.lut_ids" in inference_state
    assert all("weight" not in key for key in inference_state)

    restored = torch.nn.Sequential(
        FixedBinarization(thresholds=torch.zeros_like(thresholds)),
        torch.nn.Flatten(),
        LogicDense(
            in_dim=4,
            out_dim=6,
            parametrization="light",
            connections_kwargs={"init_method": "random-unique"},
        ),
        GroupSum(2),
    )
    load_inference_state_dict(restored, inference_state)

    actual = restored(x)
    assert torch.allclose(expected, actual.float(), atol=1e-6)


@torch.no_grad()
def test_inference_only_load_strict_reports_missing_key():
    layer = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )
    inference_state = layer.state_dict(inference_only=True)
    inference_state.pop("_inference.lut_ids")

    restored = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )

    with pytest.raises(RuntimeError, match="_inference.lut_ids"):
        restored.load_state_dict(inference_state, inference_only=True)


@torch.no_grad()
def test_inference_only_load_non_strict_reports_missing_key_without_raising():
    layer = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )
    inference_state = layer.state_dict(inference_only=True)
    inference_state.pop("_inference.connection_indices")

    restored = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )
    incompatible = restored.load_state_dict(
        inference_state,
        strict=False,
        inference_only=True,
    )

    assert "_inference.connection_indices" in incompatible.missing_keys
    assert restored.inference_only is False


@torch.no_grad()
def test_regular_state_dict_is_unchanged_by_inference_only_support():
    layer = LogicDense(
        in_dim=8,
        out_dim=12,
        parametrization="warp",
        connections_kwargs={"init_method": "random-unique"},
    )

    regular_state = layer.state_dict()
    inference_state = layer.state_dict(inference_only=True)

    assert "weight" in regular_state
    assert "_inference.lut_ids" not in regular_state
    assert "weight" not in inference_state
    assert "_inference.lut_ids" in inference_state


def test_inference_only_state_rejects_lut_rank_above_2_dense():
    layer = LogicDense(
        in_dim=4,
        out_dim=6,
        lut_rank=4,
        parametrization="warp",
        connections_kwargs={"init_method": "random"},
    )

    with pytest.raises(NotImplementedError, match="lut_rank=2"):
        layer.state_dict(inference_only=True)


def test_inference_only_state_rejects_lut_rank_above_2_conv():
    layer = LogicConv2d(
        in_dim=5,
        channels=1,
        num_kernels=2,
        tree_depth=1,
        receptive_field_size=4,
        lut_rank=4,
        parametrization="warp",
        connections_kwargs={"init_method": "random"},
    )

    with pytest.raises(NotImplementedError, match="lut_rank=2"):
        layer.state_dict(inference_only=True)
