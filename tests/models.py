"""Shared model definitions for the test suite.

Every model is an nn.Module subclass declaring two class attributes:

    input_shape   shape of one sample, without the batch dimension
    input_dtype   dtype its eval-mode forward expects

``MODELS`` is the canonical set the model-level and circuit-level tests sweep.
**Adding your class to MODELS is the only edit needed for it to inherit every
model-level property test.** Tests build a fresh instance per test with
``model_cls()``, since most of them mutate the model (set weights, switch to
export mode).

All logic layers use ``weight_init="random"``

Input builders and shared assertions live in tests/helpers.py.

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
LAYER_KWARGS = {
    "parametrization": "raw",
    "parametrization_kwargs": {"weight_init": "random"}
}


class DenseModel(nn.Sequential):
    """Plain stack of dense logic layers."""

    input_shape = (1000,)
    input_dtype = torch.float32

    def __init__(self):
        super().__init__(
            LogicDense(1000, 1000, **LAYER_KWARGS),
            LogicDense(1000, 1000, **LAYER_KWARGS),
        )


class ConvModel(nn.Sequential):
    """conv -> pool -> flatten -> dense -> dense -> GroupSum.

    8x8 input, receptive field 3 -> 6x6; pooling by 2 -> 3x3; 8 kernels, so
    8 * 3 * 3 = 72 features into the dense head.
    """

    input_shape = (3, 8, 8)
    input_dtype = torch.float32

    def __init__(self):
        super().__init__(
            LogicConv2d(in_dim=8, channels=3, num_kernels=8, receptive_field_size=3,
                        tree_depth=2, **LAYER_KWARGS),
            OrPooling2d(kernel_size=2, stride=2),
            nn.Flatten(),
            LogicDense(72, 64, **LAYER_KWARGS),
            LogicDense(64, 50, **LAYER_KWARGS),
            GroupSum(10),
        )


class ConvTransposeAE3dModel(nn.Sequential):
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

    input_shape = (2, 6, 6, 6)
    input_dtype = torch.float32

    def __init__(self):
        super().__init__(
            LogicConv3d(in_dim=6, channels=2, num_kernels=3, receptive_field_size=3,
                        tree_depth=2, stride=2, padding=1,
                        **LAYER_KWARGS),                     # -> 3 x 3^3
            LogicConvTranspose3d(in_dim=3, channels=3, num_kernels=2, receptive_field_size=3,
                                 tree_depth=2, stride=2, padding=1, output_padding=1,
                                 **LAYER_KWARGS),            # -> 2 x 6^3
            OrPooling3d(kernel_size=2, stride=2),                                # -> 2 x 3^3
            nn.Flatten(),
        )


class BinarizedDenseModel(nn.Sequential):
    """DenseModel behind a LearnableBinarization front-end.

    Its two thresholds double the feature count, so the input is half as wide.
    """

    input_shape = (500,)
    input_dtype = torch.float32

    def __init__(self):
        super().__init__(
            LearnableBinarization(thresholds=[0.33, 0.66]),
            LogicDense(1000, 1000, **LAYER_KWARGS),
            LogicDense(1000, 1000, **LAYER_KWARGS),
        )


class BinarizedConvModel(nn.Sequential):
    """ConvModel behind a per-channel LearnableBinarization front-end.

    Its two thresholds double the 3 input channels to 6, which the conv layer
    has to expect.
    """

    input_shape = (3, 8, 8)
    input_dtype = torch.float32

    def __init__(self):
        super().__init__(
            LearnableBinarization(thresholds=[0.33, 0.66], one_per="channel", feature_dim=1),
            LogicConv2d(in_dim=8, channels=6, num_kernels=8, receptive_field_size=3,
                        tree_depth=2, **LAYER_KWARGS),
            OrPooling2d(kernel_size=2, stride=2),
            nn.Flatten(),
            LogicDense(72, 64, **LAYER_KWARGS),
            LogicDense(64, 50, **LAYER_KWARGS),
            GroupSum(10),
        )


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
                                **LAYER_KWARGS)
        self.pool = OrPooling2d(kernel_size=2, stride=2)
        self.dense = LogicDense(1801, 1000, **LAYER_KWARGS)
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


# The canonical set swept by model-level and circuit-level property tests.
# Add your model class here and it inherits every one of them.
MODELS = [
    DenseModel,
    ConvModel,
    ConvTransposeAE3dModel,
    BranchModel,
    AnyLogicModel,
]

# Models actually built out of torchlogix layers. AnyLogicModel is not: it has
# no trainable parameters, and it constructs constants (ones/zeros/triu), so it
# is excluded both from gradient properties and from the "lowers to pure logic"
# property, which only our layers are expected to satisfy.
TORCHLOGIX_MODELS = [m for m in MODELS if m is not AnyLogicModel]

# Models with a binarization front-end, used where a stochastic component has
# to collapse onto its discrete counterpart.
BINARIZED_MODELS = [BinarizedDenseModel, BinarizedConvModel, BranchModel]
