"""Shared model definitions for the test suite.

Every model exposes ``input_shape`` (spatial/feature shape, no batch dim), so a
test can build an input for any model without a hand-maintained lookup table.

``MODELS`` is the canonical set that the model-level and circuit-level tests
sweep. **Adding a model there is the only edit needed for it to inherit every
model-level property test.**

All logic layers use ``weight_init="random"``: residual init only improves
training dynamics, so random is the harder case to get right.

This module is imported, not collected - pytest only collects ``test_*.py``.
"""
import math

import pytest
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


def _pooled_flat_size(conv, pool_kernel=2, pool_stride=2):
    """Flattened feature count after `conv` followed by an Or-pooling layer.

    Derived from conv.kernel_positions rather than by running a forward pass:
    a probe forward through a logic layer would consume RNG and change the
    initialization of every layer built afterwards.
    """
    pooled = [(p - pool_kernel) // pool_stride + 1 for p in conv.kernel_positions]
    return conv.num_kernels * math.prod(pooled)


class DenseModel(nn.Sequential):
    """Plain sequential stack of dense logic layers.

    With ``binarize=True`` a LearnableBinarization front-end is prepended; its
    two thresholds double the feature count, so the input is half as wide.
    """

    input_dtype = torch.float32

    def __init__(self, binarize=False):
        width = 1000
        layers = []
        if binarize:
            layers.append(LearnableBinarization(thresholds=[0.33, 0.66]))
        layers += [
            LogicDense(width, width, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
            LogicDense(width, width, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        ]
        super().__init__(*layers)
        self.input_shape = (width // 2,) if binarize else (width,)


class ConvModel(nn.Sequential):
    """Sequential conv stack: conv -> pool -> flatten -> dense -> dense [-> GroupSum].

    ``ndim`` picks the 2D or 3D layer family, so the 2D and 3D variants stay
    one definition instead of two that drift apart. ``group_sum=False`` gives
    the bare-logic-output variant. With ``binarize=True`` a per-channel
    LearnableBinarization front-end doubles the channel count.
    """

    input_dtype = torch.float32

    # (hidden, out, groups) per dimension. FixedDenseConnections requires
    # out_dim * lut_rank >= in_dim to cover all inputs, and the 3D stack
    # flattens to 216 features against the 2D stack's 72, so the two need
    # different widths. Values match the models these replaced.
    _DENSE_SHAPE = {2: (64, 50, 10), 3: (128, 64, 8)}

    def __init__(self, ndim=2, group_sum=True, binarize=False):
        assert ndim in (2, 3)
        in_dim, channels, num_kernels = 8, 3, 8
        hidden, out_features, groups = self._DENSE_SHAPE[ndim]
        conv_cls = LogicConv2d if ndim == 2 else LogicConv3d
        pool_cls = OrPooling2d if ndim == 2 else OrPooling3d

        conv = conv_cls(
            in_dim=in_dim,
            channels=channels * 2 if binarize else channels,
            num_kernels=num_kernels,
            receptive_field_size=3,
            tree_depth=2,
            parametrization_kwargs=RANDOM_INIT,
        )
        n_flat = _pooled_flat_size(conv)

        layers = []
        if binarize:
            layers.append(
                LearnableBinarization(thresholds=[0.33, 0.66], one_per="channel", feature_dim=1)
            )
        layers += [
            conv,
            pool_cls(kernel_size=2, stride=2),
            nn.Flatten(),
            LogicDense(n_flat, hidden, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
            LogicDense(hidden, out_features, parametrization="raw", parametrization_kwargs=RANDOM_INIT),
        ]
        if group_sum:
            layers.append(GroupSum(groups))
        super().__init__(*layers)
        self.input_shape = (channels, *([in_dim] * ndim))


class BranchModel(nn.Module):
    """Custom forward: an image path and a scalar-feature path, recombined.

    The flat input carries the image and one extra feature, so the model has to
    split, reshape, and concatenate - which a plain Sequential cannot express.
    """

    input_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.img_shape = (3, 32, 32)
        n_img = math.prod(self.img_shape)
        self.conv = LogicConv2d(
            in_dim=32, channels=3, num_kernels=8, receptive_field_size=3,
            tree_depth=2, parametrization_kwargs=RANDOM_INIT,
        )
        self.pool = OrPooling2d(kernel_size=2, stride=2)
        n_flat = _pooled_flat_size(self.conv)
        self.dense = LogicDense(
            n_flat + 1, 1000, parametrization="raw", parametrization_kwargs=RANDOM_INIT
        )
        self.group_sum = GroupSum(10)
        self.input_shape = (n_img + 1,)

    def forward(self, x):
        assert x.shape[1:] == self.input_shape
        img, feat = x[:, :-1].reshape(-1, *self.img_shape), x[:, -1:]
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x


class ConvTransposeAE3dModel(nn.Sequential):
    """3D autoencoder: LogicConv3d halves 6^3 to 3^3, LogicConvTranspose3d restores it.

    Both layers use nonzero padding, and the decoder a nonzero output_padding,
    so this one model covers the whole transposed-conv export path:

    * input dilation (stride > 1) expressed functionally - building a zero
      tensor and writing into it makes constant_fold_views reject the graph
    * padding applied inside FixedConvTransposeConnections rather than by the
      layer, which would otherwise double-pad
    * kernel_positions describing the larger transposed output
    """

    input_dtype = torch.float32

    def __init__(self):
        super().__init__(
            LogicConv3d(in_dim=6, channels=2, num_kernels=3, receptive_field_size=3,
                        tree_depth=2, stride=2, padding=1,
                        parametrization_kwargs=RANDOM_INIT),                    # -> 3 x 3^3
            LogicConvTranspose3d(in_dim=3, channels=3, num_kernels=2, receptive_field_size=3,
                                 tree_depth=2, stride=2, padding=1, output_padding=1,
                                 parametrization_kwargs=RANDOM_INIT),           # -> 2 x 6^3
            nn.Flatten(),   # Circuit outputs are flat; the spatial round-trip
                            # itself is asserted in test_layers.py
        )
        self.input_shape = (2, 6, 6, 6)


class AnyLogicModel(nn.Module):
    """Assorted non-torchlogix logic and reshaping ops, to test from_model's reach.

    Deliberately contains no torchlogix layer at all, and mixes boolean outputs
    with summed (reduction) outputs. Recovered from 5b40074, with the constant
    mask rebuilt via torch.cat instead of `mask[4:, :] = 0`: mutating a freshly
    created constant in place is untraceable, and InPlaceConstMutationModel
    below is the test that such a model is *rejected*.
    """

    input_dtype = torch.bool

    def __init__(self):
        super().__init__()
        self.input_shape = (4, 8, 8)

    def forward(self, x):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        x1 = torch.flip(x1, dims=[1])       # flip rows
        x2 = x2.permute(0, 2, 1)            # swap H and W

        # Zero the bottom half of x3. Built in one expression, not mutated.
        ones = torch.ones(4, 8, dtype=x.dtype, device=x.device)
        mask = torch.cat([ones, torch.zeros(4, 8, dtype=x.dtype, device=x.device)], dim=0)
        x3 = x3 & mask

        # Keep only the upper triangle of x4.
        tri = torch.triu(torch.ones(8, 8, dtype=x.dtype, device=x.device))
        x4 = x4 & tri

        out = x1 | ((x2 & x3) ^ x4)
        out = out.flatten(1)                # (B, 64)

        # Mixed output kinds: two reductions plus a boolean slice. Do NOT cast
        # out3 to match - an explicit .to(dtype) here stops from_model
        # recognising the outputs at all and yields a zero-output circuit.
        out1 = out[:, :8].sum(dim=1, keepdim=True)
        out2 = out[:, 8:16].sum(dim=1, keepdim=True)
        out3 = out[:, 16:]
        return torch.cat([out1, out2, out3], dim=1)


class InPlaceConstMutationModel(nn.Module):
    """Mutates a constant tensor in place after creation.

    `mask = torch.ones(8, 8); mask[4:, :] = 0` cannot be constant-folded or
    safely traced by torch.fx, so both Circuit.from_model (see
    constant_fold_views / _reject_orphaned_impure_ops in circuit.py) and the
    alkaid plugin (_fold_constant_views in _alkaid_plugin.py) must reject it
    clearly rather than silently building a wrong circuit.

    Deliberately NOT in MODELS - it exists to be rejected.
    """

    input_dtype = torch.bool

    def __init__(self):
        super().__init__()
        self.input_shape = (8, 8)

    def forward(self, x):
        mask = torch.ones(8, 8, dtype=x.dtype, device=x.device)
        mask[4:, :] = 0
        return x & mask


# The canonical set swept by model-level and circuit-level property tests.
# Add a model here and it inherits every one of them.
MODELS = [
    DenseModel,
    ConvModel,
    BranchModel,
    ConvTransposeAE3dModel,
    AnyLogicModel,
]

# Models actually built out of torchlogix layers. AnyLogicModel is not: it has
# no trainable parameters, and it constructs constants (ones/zeros/triu), so it
# is excluded both from gradient properties and from the "lowers to pure logic"
# FX-purity property, which only our layers are expected to satisfy.
TORCHLOGIX_MODELS = [m for m in MODELS if m is not AnyLogicModel]

# Extra ConvModel configurations worth exporting, beyond the canonical set:
# the 3D layer family, and the variant whose output is raw logic rather than
# GroupSum scores. These are configurations, not separate definitions.
EXPORT_VARIANTS = [
    pytest.param(lambda: ConvModel(ndim=3), id="ConvModel-3d"),
    pytest.param(lambda: ConvModel(group_sum=False), id="ConvModel-no-groupsum"),
    pytest.param(lambda: ConvModel(ndim=3, group_sum=False), id="ConvModel-3d-no-groupsum"),
]


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
