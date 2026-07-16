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
# Model fixtures (shared across test_export_mode.py and test_alkaid_plugin.py)
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
        LogicConv2d(in_dim=8, channels=3, num_kernels=7, receptive_field_size=3, tree_depth=2),
        OrPooling2d(kernel_size=2, stride=2),
        nn.Flatten(),          # 3 × 3 × 8 = 72
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


# ---------------------------------------------------------------------------
# Input fixtures
# ---------------------------------------------------------------------------

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
