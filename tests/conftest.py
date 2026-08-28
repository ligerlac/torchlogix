import os

# torch and scikit-learn (a required torchlogix dependency) each bundle their
# own separate copy of the OpenMP runtime (libomp.dylib on macOS). Loading
# both into one process trips OpenMP's duplicate-runtime safety check and
# aborts the interpreter. This is the standard, low-risk workaround for that
# specific benign double-load (not a general-purpose crash suppressant) -
# must be set before torch/sklearn are actually imported. setdefault() so an
# explicit value set by the caller isn't clobbered.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pytest

from models import (
    AnyLogicModel,
    BranchModel,
    ConvModel,
    ConvTransposeAE3dModel,
    DenseModel,
)

# ---------------------------------------------------------------------------
# Model fixtures
#
# The model definitions themselves live in tests/models.py, which is also
# where MODELS lists the canonical set. Tests that sweep every model should
# parametrize over MODELS directly; these fixtures are for tests that just
# want one ready-made instance.
#
# Every model exposes `input_shape`; build an input with
# models.random_bool_input(model) rather than a matching input fixture.
# ---------------------------------------------------------------------------

@pytest.fixture
def dense_model():
    model = DenseModel()
    model.eval()
    return model


@pytest.fixture
def conv_model():
    model = ConvModel()
    model.eval()
    return model


@pytest.fixture
def conv3d_model():
    model = ConvModel(ndim=3)
    model.eval()
    return model


@pytest.fixture
def branch_model():
    model = BranchModel()
    model.eval()
    return model


@pytest.fixture
def conv_transpose_ae_model():
    model = ConvTransposeAE3dModel()
    model.eval()
    return model


@pytest.fixture
def any_logic_model():
    model = AnyLogicModel()
    model.eval()
    return model
