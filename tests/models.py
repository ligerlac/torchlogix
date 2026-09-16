"""Shared model definitions for the test suite.

Each model is a plain factory function that builds and returns it. Every model
carries two attributes:

    input_shape   shape of one sample, without the batch dimension
    input_dtype   dtype its eval-mode forward expects

``MODELS`` is the canonical set the model-level and circuit-level tests sweep.
**Adding your function to MODELS is the only edit needed for it to inherit
every model-level property test.**

All logic layers use ``weight_init="random"``: residual init only improves
training dynamics, so random is the harder case to get right.

This module is imported, not collected - pytest only collects ``test_*.py``.
"""
import torch
import torch.nn as nn

from torchlogix.layers import (
    GroupSum,
    LearnableBinarization,
    LogicConv2d,
    LogicConv3d,
    LogicConvTranspose3d,
    LogicDense,
    OrPooling2d,
    OrPooling3d,
)

# Residual init only helps training dynamics; random is the harder case.
RANDOM_INIT = {"weight_init": "random"}


def dense_model():
    """Plain stack of dense logic layers."""
    model = nn.Sequential(
        LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
    )
    model.input_shape = (1000,)
    model.input_dtype = torch.float32
    return model


def conv_model():
    """conv -> pool -> flatten -> dense -> dense -> GroupSum.

    8x8 input, receptive field 3 -> 6x6; pooling by 2 -> 3x3; 8 kernels, so
    8 * 3 * 3 = 72 features into the dense head.
    """
    model = nn.Sequential(
        LogicConv2d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3,
                    tree_depth=2, parametrization_kwargs=RANDOM_INIT),
        OrPooling2d(kernel_size=2, stride=2),
        nn.Flatten(),
        LogicDense(72, 64, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        LogicDense(64, 50, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        GroupSum(10),
    )
    model.input_shape = (3, 8, 8)
    model.input_dtype = torch.float32
    return model


def conv_transpose_ae_model():
    """3D autoencoder: conv halves 6^3 to 3^3, transposed conv restores it.

    Both layers pad, and the decoder also uses output_padding, so this one
    model covers the whole transposed-conv export path: strided input
    dilation, padding applied inside the connections rather than by the layer,
    and kernel_positions describing the enlarged output.

    The trailing OrPooling3d is this suite's only 3D pooling in export mode -
    without it the boolean unfold-and-OR kernel in OrPooling3d goes untested.
    Flattened last because Circuit outputs are flat; the spatial round trip
    itself is asserted in test_layers.py.
    """
    model = nn.Sequential(
        LogicConv3d(in_dim=6, channels=2, num_kernels=3, receptive_field_size=3,
                    tree_depth=2, stride=2, padding=1,
                    parametrization_kwargs=RANDOM_INIT),                     # -> 3 x 3^3
        LogicConvTranspose3d(in_dim=3, channels=3, num_kernels=2, receptive_field_size=3,
                             tree_depth=2, stride=2, padding=1, output_padding=1,
                             parametrization_kwargs=RANDOM_INIT),            # -> 2 x 6^3
        OrPooling3d(kernel_size=2, stride=2),                                # -> 2 x 3^3
        nn.Flatten(),
    )
    model.input_shape = (2, 6, 6, 6)
    model.input_dtype = torch.float32
    return model


def binarized_dense_model():
    """dense_model behind a LearnableBinarization front-end.

    Its two thresholds double the feature count, so the input is half as wide.
    """
    model = nn.Sequential(
        LearnableBinarization(thresholds=[0.33, 0.66]),
        LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
    )
    model.input_shape = (500,)
    model.input_dtype = torch.float32
    return model


def binarized_conv_model():
    """conv_model behind a per-channel LearnableBinarization front-end.

    Its two thresholds double the 3 input channels to 6, which the conv layer
    has to expect.
    """
    model = nn.Sequential(
        LearnableBinarization(thresholds=[0.33, 0.66], one_per="channel", feature_dim=1),
        LogicConv2d(in_dim=8, channels=6, num_kernels=8, receptive_field_size=3,
                    tree_depth=2, parametrization_kwargs=RANDOM_INIT),
        OrPooling2d(kernel_size=2, stride=2),
        nn.Flatten(),
        LogicDense(72, 64, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        LogicDense(64, 50, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        GroupSum(10),
    )
    model.input_shape = (3, 8, 8)
    model.input_dtype = torch.float32
    return model


class BranchModel(nn.Module):
    """Custom forward: an image path and a scalar-feature path, recombined.

    The flat input carries a 3x32x32 image plus one extra feature, so the
    model has to split, reshape and concatenate - which a plain Sequential
    cannot express. The conv gives 30x30, pooling by 2 gives 15x15, and with
    8 kernels that is 1800 features, plus the extra one.
    """

    input_shape = (3 * 32 * 32 + 1,)
    input_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=32, channels=3, num_kernels=8,
                                receptive_field_size=3, tree_depth=2,
                                parametrization_kwargs=RANDOM_INIT)
        self.pool = OrPooling2d(kernel_size=2, stride=2)
        self.dense = LogicDense(1801, 1000, parametrization="raw",
                                parametrization_kwargs=RANDOM_INIT)
        self.group_sum = GroupSum(10)

    def forward(self, x):
        img, feat = x[:, :-1].reshape(-1, 3, 32, 32), x[:, -1:]
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x


class AnyLogicModel(nn.Module):
    """Assorted non-torchlogix logic and reshaping ops, to test from_model's reach.

    Deliberately contains no torchlogix layer at all, and mixes boolean
    outputs with summed (reduction) outputs.

    The constant mask is built with torch.cat rather than `mask[4:, :] = 0`:
    mutating a freshly created constant in place is untraceable, which is
    exactly what InPlaceConstMutationModel below exists to demonstrate.
    """

    input_shape = (4, 8, 8)
    input_dtype = torch.bool

    def forward(self, x):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        x1 = torch.flip(x1, dims=[1])       # flip rows
        x2 = x2.permute(0, 2, 1)            # swap H and W

        # Zero the bottom half of x3. Built in one expression, not mutated.
        ones = torch.ones(4, 8, dtype=x.dtype, device=x.device)
        zeros = torch.zeros(4, 8, dtype=x.dtype, device=x.device)
        x3 = x3 & torch.cat([ones, zeros], dim=0)

        # Keep only the upper triangle of x4.
        x4 = x4 & torch.triu(torch.ones(8, 8, dtype=x.dtype, device=x.device))

        out = (x1 | ((x2 & x3) ^ x4)).flatten(1)        # (B, 64)

        # Mixed output kinds: two reductions plus a boolean slice. Do NOT cast
        # the slice to match - an explicit .to(dtype) here stops from_model
        # recognising the outputs at all and yields a zero-output circuit.
        out1 = out[:, :8].sum(dim=1, keepdim=True)
        out2 = out[:, 8:16].sum(dim=1, keepdim=True)
        return torch.cat([out1, out2, out[:, 16:]], dim=1)


class InPlaceConstMutationModel(nn.Module):
    """Mutates a constant tensor in place after creation.

    `mask = torch.ones(8, 8); mask[4:, :] = 0` cannot be constant-folded or
    safely traced by torch.fx, so both Circuit.from_model and the alkaid
    plugin must reject it clearly rather than silently building a wrong
    circuit.

    Deliberately NOT in MODELS - it exists to be rejected.
    """

    input_shape = (8, 8)
    input_dtype = torch.bool

    def forward(self, x):
        mask = torch.ones(8, 8, dtype=x.dtype, device=x.device)
        mask[4:, :] = 0
        return x & mask


def branch_model():
    return BranchModel()


def any_logic_model():
    return AnyLogicModel()


def in_place_const_mutation_model():
    return InPlaceConstMutationModel()


# The canonical set swept by model-level and circuit-level property tests.
# Add your model function here and it inherits every one of them.
MODELS = [
    dense_model,
    conv_model,
    conv_transpose_ae_model,
    branch_model,
    any_logic_model,
]

# Models actually built out of torchlogix layers. any_logic_model is not: it
# has no trainable parameters, and it constructs constants (ones/zeros/triu),
# so it is excluded both from gradient properties and from the "lowers to pure
# logic" property, which only our layers are expected to satisfy.
TORCHLOGIX_MODELS = [m for m in MODELS if m is not any_logic_model]

# Models with a binarization front-end, used where a stochastic component has
# to collapse onto its discrete counterpart.
BINARIZED_MODELS = [binarized_dense_model, binarized_conv_model, branch_model]


def random_bool_input(model, batch_size=1, seed=None):
    """Random boolean input matching `model.input_shape`.

    Export-mode and circuit tests always want bool, whatever the model's
    eval-mode dtype is.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, 2, (batch_size, *model.input_shape), dtype=torch.bool)


def model_input(model, batch_size=1, seed=None):
    """Random input in the dtype `model`'s eval-mode forward expects.

    Logic layers do float arithmetic when not in export mode, while the
    pure-bitwise models need an integral dtype - hence `input_dtype`.
    """
    x = random_bool_input(model, batch_size=batch_size, seed=seed)
    return x if model.input_dtype == torch.bool else x.to(model.input_dtype)
