import ctypes
import math
import shutil
import subprocess
import tempfile
import time
from typing import Union, List, Dict, Any

import numpy as np
import numpy.typing
import torch

from .layers.conv import LogicConv2d, LogicConv3d, OrPooling
from .layers.dense import LogicDense
from .layers.groupsum import GroupSum
from .layers.thresholding import LearnableThermometerThresholding

ALL_OPERATIONS = [
    "zero", "and", "not_implies", "a", "not_implied_by", "b", "xor", "or",
    "not_or", "not_xor", "not_b", "implied_by", "not_a", "implies", "not_and", "one",
]

BITS_TO_DTYPE = {1: "bool", 8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {1: "0", 8: "(char) 0", 16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {1: "1", 8: "(char) 1", 16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {
    1: ctypes.c_bool, 8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64,
}
BITS_TO_NP_DTYPE = {1: np.bool_, 8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


class CompiledLogicNet(torch.nn.Module):
    """
    Unified compiled logic network that handles convolutional, pooling, and linear layers.
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        input_shape: tuple,
        device: str = "cpu",
        num_bits: int = 64,
        cpu_compiler: str = "gcc",
        verbose: bool = False,
        use_bitpacking: bool = True,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.apply_thresholding = False  # Will be set during _parse_model
        self.use_bitpacking = use_bitpacking  # Store user's choice
        self.cpu_compiler = cpu_compiler

        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [1, 8, 16, 32, 64]
        if num_bits > 1 and not use_bitpacking:
            raise ValueError("num_bits > 1 requires use_bitpacking=True.")

        # Initialize layer storage
        self.thresholding_layer = None
        self.conv_layers = []
        self.pooling_layers = []
        self.linear_layers = []
        self.num_classes = None
        self.input_shape = input_shape
        self.layer_order = []
        self.lib_fn = None

        # GroupSum information
        self.tau = 1
        self.beta = 0

        if model is not None:
            self._parse_model(verbose)

    def _parse_model(self, verbose: bool):
        """Parse the model structure, handling conv, pooling, and linear layers."""

        # Detect if first layer is thresholding
        if len(self.model) > 0 and isinstance(self.model[0], LearnableThermometerThresholding):
            self.apply_thresholding = True

            # Validate bitpacking constraint
            if self.use_bitpacking:
                raise ValueError(
                    "Bitpacking (use_bitpacking=True) cannot be used with thresholding layers. "
                    "The thresholding layer requires processing one sample at a time."
                )

            # Set constraints for thresholding mode
            self.num_bits = 1
            self.use_bitpacking = False

        # Find GroupSum layer for num_classes
        for layer in self.model:
            if isinstance(layer, GroupSum):
                self.num_classes = layer.k
                self.tau = layer.tau
                self.beta = layer.beta
                break

        if self.num_classes is None:
            raise ValueError("No GroupSum layer found in model")

        # Parse all layers and track execution order
        for layer in self.model:
            if isinstance(layer, LearnableThermometerThresholding):
                if len(self.layer_order) > 0:
                    raise ValueError("LearnableThermometerThresholding layer must appear first in layer order.")
                thresholding_info = self._extract_thresholding_layer_info(layer)
                self.thresholding_layer = thresholding_info  # there will only be one thresholding layer
            elif isinstance(layer, LogicConv2d):
                conv_info = self._extract_conv_layer_info(layer)
                self.conv_layers.append(conv_info)
                self.layer_order.append(('conv', len(self.conv_layers) - 1))
            elif isinstance(layer, LogicConv3d):
                conv_info = self._extract_conv_layer_info(layer)
                self.conv_layers.append(conv_info)
                self.layer_order.append(('conv', len(self.conv_layers) - 1))
            elif isinstance(layer, OrPooling):
                pool_info = self._extract_pooling_layer_info(layer)
                self.pooling_layers.append(pool_info)
                self.layer_order.append(('pool', len(self.pooling_layers) - 1))
            elif isinstance(layer, LogicDense):
                self.linear_layers.append(
                    (layer.indices[0], layer.indices[1], layer.weight.argmax(1))
                )
                self.layer_order.append(('linear', len(self.linear_layers) - 1))
            elif isinstance(layer, torch.nn.Flatten):
                self.layer_order.append(('flatten', 0))
                if verbose:
                    print(f"Found Flatten layer")
            elif isinstance(layer, GroupSum):
                if verbose:
                    print(f"Found GroupSum layer (tau={self.tau}, beta={self.beta}) with {layer.k} classes")
            else:
                if verbose:
                    print(f"Warning: Unknown layer type: {type(layer)}")

        if verbose:
            print((f"Parsed {int(self.thresholding_layer is not None)} thresholding, {len(self.conv_layers)} conv," 
                   f"{len(self.pooling_layers)} pooling, {len(self.linear_layers)} linear layers"))
            print(f"Layer execution order: {self.layer_order}")

        # Validate model structure
        if not self.conv_layers and not self.linear_layers:
            raise ValueError("Model must contain at least one LogicConv2d, LogicConv3d, or LogicDense layer.")

    def _extract_thresholding_layer_info(self, layer: LearnableThermometerThresholding) -> Dict[str, Any]:
        """Extract information from a thresholding layer for compilation."""
        return {
            'num_thresholds': layer.num_thresholds,
            'thresholds': layer.get_thresholds(),
        }

    def _extract_conv_layer_info(self, layer: Union[LogicConv2d, LogicConv3d]) -> Dict[str, Any]:
        """Extract information from a LogicConv2d or LogicConv3d layer for compilation."""
        tree_operations = []
        for level_idx, level_weights in enumerate(layer.tree_weights):
            level_ops = []
            for weight_param in level_weights:
                ops = weight_param.argmax(1).cpu().numpy()
                level_ops.append(ops)
            tree_operations.append(level_ops)

        return {
            'indices': layer.indices,
            'tree_operations': tree_operations,
            'tree_depth': layer.tree_depth,
            'num_kernels': layer.num_kernels,
            'in_dim': layer.in_dim,
            'receptive_field_size': layer.receptive_field_size,
            'stride': layer.stride,
            'padding': layer.padding,
            'channels': layer.channels,
        }

    def _extract_pooling_layer_info(self, layer: OrPooling) -> Dict[str, Any]:
        """Extract information from an OrPooling layer for compilation."""
        return {
            'kernel_size': layer.kernel_size,
            'stride': layer.stride,
            'padding': layer.padding,
        }

    def get_gate_code(self, var1: str, var2: str, gate_op: int) -> str:
        """Generate C code for a logic gate operation."""
        operation_name = ALL_OPERATIONS[gate_op]

        if self.num_bits > 1:
            if operation_name == "zero":
                res = BITS_TO_ZERO_LITERAL[self.num_bits]
            elif operation_name == "and":
                res = f"{var1} & {var2}"
            elif operation_name == "not_implies":
                res = f"{var1} & ~{var2}"
            elif operation_name == "a":
                res = f"{var1}"
            elif operation_name == "not_implied_by":
                res = f"{var2} & ~{var1}"
            elif operation_name == "b":
                res = f"{var2}"
            elif operation_name == "xor":
                res = f"{var1} ^ {var2}"
            elif operation_name == "or":
                res = f"{var1} | {var2}"
            elif operation_name == "not_or":
                res = f"~({var1} | {var2})"
            elif operation_name == "not_xor":
                res = f"~({var1} ^ {var2})"
            elif operation_name == "not_b":
                res = f"~{var2}"
            elif operation_name == "implied_by":
                res = f"~{var2} | {var1}"
            elif operation_name == "not_a":
                res = f"~{var1}"
            elif operation_name == "implies":
                res = f"~{var1} | {var2}"
            elif operation_name == "not_and":
                res = f"~({var1} & {var2})"
            elif operation_name == "one":
                res = f"~{BITS_TO_ZERO_LITERAL[self.num_bits]}"
            else:
                raise ValueError(f"Operator {operation_name} unknown.")

            if self.num_bits == 8:
                res = f"(char) ({res})"
            elif self.num_bits == 16:
                res = f"(short) ({res})"

        else:
            if operation_name == "zero":
                res = BITS_TO_ZERO_LITERAL[self.num_bits]
            elif operation_name == "and":
                res = f"{var1} && {var2}"
            elif operation_name == "not_implies":
                res = f"{var1} && !{var2}"
            elif operation_name == "a":
                res = f"{var1}"
            elif operation_name == "not_implied_by":
                res = f"{var2} && !{var1}"
            elif operation_name == "b":
                res = f"{var2}"
            elif operation_name == "xor":
                res = f"{var1} != {var2}"
            elif operation_name == "or":
                res = f"{var1} || {var2}"
            elif operation_name == "not_or":
                res = f"!({var1} || {var2})"
            elif operation_name == "not_xor":
                res = f"({var1} == {var2})"
            elif operation_name == "not_b":
                res = f"!{var2}"
            elif operation_name == "implied_by":
                res = f"(!{var2}) || {var1}"
            elif operation_name == "not_a":
                res = f"!{var1}"
            elif operation_name == "implies":
                res = f"(!{var1}) || {var2}"
            elif operation_name == "not_and":
                res = f"!({var1} && {var2})"
            elif operation_name == "one":
                res = f"~{BITS_TO_ZERO_LITERAL[self.num_bits]}"
            else:
                raise ValueError(f"Operator {operation_name} unknown.")

        return res

    def _calculate_layer_output_sizes_and_shapes(self) -> List[tuple]:
        """Calculate output sizes and shapes for all layers in execution order."""
        layer_info = []
        if self.thresholding_layer is None:
            current_shape = self.input_shape
        else:
            thresholding_info = self.thresholding_layer
            if len(self.input_shape)==2:
                current_shape = (thresholding_info['num_thresholds'], self.input_shape[0], self.input_shape[1])
            elif len(self.input_shape)==3:
                if self.input_shape[0]!=1:
                    raise ValueError("If the input tensor is rank 3, index 0 must be a dummy index.")
                current_shape = (thresholding_info['num_thresholds'], self.input_shape[1], self.input_shape[2])
            else:
                raise ValueError("The thresholding layer can only accept input tensors of rank 2 or 3.")

        for layer_type, layer_idx in self.layer_order:
            if layer_type == 'conv':
                conv_info = self.conv_layers[layer_idx]
                if len(current_shape) == 3:  # (C, H, W)
                    c, h, w = current_shape
                    h_out = ((h + 2 * conv_info['padding'] - conv_info['receptive_field_size'])
                             // conv_info['stride']) + 1
                    w_out = ((w + 2 * conv_info['padding'] - conv_info['receptive_field_size'])
                             // conv_info['stride']) + 1
                    output_shape = (conv_info['num_kernels'], h_out, w_out)
                    output_size = conv_info['num_kernels'] * h_out * w_out
                elif len(current_shape) == 4: # (C, H, W, D)
                    c, h, w, d = current_shape
                    h_out = ((h + 2 * conv_info['padding'] - conv_info['receptive_field_size'][0])
                             // conv_info['stride']) + 1
                    w_out = ((w + 2 * conv_info['padding'] - conv_info['receptive_field_size'][1])
                             // conv_info['stride']) + 1
                    d_out = ((d + 2 * conv_info['padding'] - conv_info['receptive_field_size'][2])
                             // conv_info['stride']) + 1
                    output_shape = (conv_info['num_kernels'], h_out, w_out, d_out)
                    output_size = conv_info['num_kernels'] * h_out * w_out * d_out
                else:
                    raise ValueError(f"Conv layer expects 3D or 4D input, got {len(current_shape)}D")

            elif layer_type == 'pool':
                pool_info = self.pooling_layers[layer_idx]
                if len(current_shape) == 3:  # (C, H, W)
                    c, h, w = current_shape
                    h_out = ((h + 2 * pool_info['padding'] - pool_info['kernel_size']) 
                             // pool_info['stride']) + 1
                    w_out = ((w + 2 * pool_info['padding'] - pool_info['kernel_size']) 
                             // pool_info['stride']) + 1
                    output_shape = (c, h_out, w_out)
                    output_size = c * h_out * w_out
                elif len(current_shape) == 4: # (C, H, W, D)
                    c, h, w, d = current_shape
                    h_out = ((h + 2 * pool_info['padding'] - pool_info['kernel_size'])
                             // pool_info['stride']) + 1
                    w_out = ((w + 2 * pool_info['padding'] - pool_info['kernel_size'])
                             // pool_info['stride']) + 1
                    d_out = ((d + 2 * pool_info['padding'] - pool_info['kernel_size'])
                             // pool_info['stride']) + 1
                    output_shape = (c, h_out, w_out, d_out)
                    output_size = c * h_out * w_out * d_out
                else:
                    raise ValueError(f"Pool layer expects 3D or 4D input, got {len(current_shape)}D")

            elif layer_type == 'flatten':
                if (len(current_shape) == 3) or (len(current_shape)==4):
                    output_size = np.prod(current_shape)
                    output_shape = (output_size,)
                else:
                    output_size = current_shape[0]
                    output_shape = current_shape

            elif layer_type == 'linear':
                layer_a, layer_b, layer_op = self.linear_layers[layer_idx]
                output_size = len(layer_a)
                output_shape = (output_size,)

            layer_info.append((layer_type, layer_idx, output_shape, output_size))
            current_shape = output_shape

        return layer_info

    def _get_conv_layer_code(self, conv_info: Dict[str, Any], layer_name: str) -> List[str]:
        """Generate C code for a convolutional layer."""
        code = []
        indices = conv_info['indices']
        tree_ops = conv_info['tree_operations']

        if len(conv_info['in_dim']) == 2:
            conv_dim = 2
            h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size'])
                    // conv_info['stride']) + 1
            w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size'])
                    // conv_info['stride']) + 1
        elif len(conv_info['in_dim']) == 3:
            conv_dim = 3
            h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][0])
                    // conv_info['stride']) + 1
            w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][1])
                    // conv_info['stride']) + 1
            d_out = ((conv_info['in_dim'][2] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][2])
                    // conv_info['stride']) + 1
        else:
            raise ValueError(f"Conv layer expects 3D or 4D input, got {len(conv_info['in_dim'])}")

        code.append(f"\t// Convolutional layer {layer_name}")

        for kernel_idx in range(conv_info['num_kernels']):
            iter_range = h_out * w_out
            if conv_dim == 3:
                iter_range *= d_out
            for pos_idx in range(iter_range):
                # First level: process receptive field positions
                level_0_indices = indices[0]
                left_indices = level_0_indices[0][kernel_idx, pos_idx]
                right_indices = level_0_indices[1][kernel_idx, pos_idx]

                # Generate variables for the first level
                for gate_idx in range(2**conv_info['tree_depth']):
                    if conv_dim == 2:
                        left_h, left_w, left_c = left_indices[gate_idx]
                        right_h, right_w, right_c = right_indices[gate_idx]
                    else:
                        left_h, left_w, left_d, left_c = left_indices[gate_idx]
                        right_h, right_w, right_d, right_c = right_indices[gate_idx]

                    gate_op = tree_ops[0][gate_idx][kernel_idx]

                    # Determine input source
                    prev_layer_name = self._get_previous_layer_name(layer_name)
                    if prev_layer_name == "inp":
                        if conv_dim==2:
                            left_var = (
                                f"inp[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + "
                                f"{left_h} * {conv_info['in_dim'][1]} + {left_w}]"
                            )
                            right_var = (
                                f"inp[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + "
                                f"{right_h} * {conv_info['in_dim'][1]} + {right_w}]"
                            )
                        else:
                            left_var = (
                                f"inp[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{left_h} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{left_w} * {conv_info['in_dim'][2]} + {left_d}]"
                            )
                            right_var = (
                                f"inp[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{right_h} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{right_w} * {conv_info['in_dim'][2]} + {right_d}]"
                            )
                    else:
                        if conv_dim == 2:
                            left_var = (
                                f"layer_{prev_layer_name}_out[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + "
                                f"{left_h} * {conv_info['in_dim'][1]} + {left_w}]"
                            )
                            right_var = (
                                f"layer_{prev_layer_name}_out[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + "
                                f"{right_h} * {conv_info['in_dim'][1]} + {right_w}]"
                            )
                        else:
                            left_var = (
                                f"layer_{prev_layer_name}_out[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{left_h} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{left_w} * {conv_info['in_dim'][2]} + {left_d}]"
                            )
                            right_var = (
                                f"layer_{prev_layer_name}_out[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{right_h} * {conv_info['in_dim'][1]} * {conv_info['in_dim'][2]} + "
                                f"{right_w} * {conv_info['in_dim'][2]} + {right_d}]"
                            )


                    var_name = f"conv_{layer_name}_k{kernel_idx}_p{pos_idx}_l0_g{gate_idx}"
                    code.append(
                        f"\tconst {BITS_TO_DTYPE[self.num_bits]} {var_name} = "
                        f"{self.get_gate_code(left_var, right_var, gate_op)};"
                    )

                # Process remaining tree levels
                for level in range(1, conv_info['tree_depth'] + 1):
                    level_indices = indices[level]
                    left_gate_indices = level_indices[0]
                    right_gate_indices = level_indices[1]

                    num_gates = len(left_gate_indices)
                    for gate_idx in range(num_gates):
                        left_idx = left_gate_indices[gate_idx]
                        right_idx = right_gate_indices[gate_idx]

                        gate_op = tree_ops[level][gate_idx][kernel_idx]

                        left_var = f"conv_{layer_name}_k{kernel_idx}_p{pos_idx}_l{level-1}_g{left_idx}"
                        right_var = f"conv_{layer_name}_k{kernel_idx}_p{pos_idx}_l{level-1}_g{right_idx}"

                        if level == conv_info['tree_depth']:
                            if conv_dim == 2:
                                output_idx = (kernel_idx * h_out * w_out + pos_idx)
                            else:
                                output_idx = (kernel_idx * h_out * w_out * d_out + pos_idx)
                            code.append(
                                f"\tlayer_{layer_name}_out[{output_idx}] = "
                                f"{self.get_gate_code(left_var, right_var, gate_op)};"
                            )
                        else:
                            var_name = f"conv_{layer_name}_k{kernel_idx}_p{pos_idx}_l{level}_g{gate_idx}"
                            code.append(
                                f"\tconst {BITS_TO_DTYPE[self.num_bits]} {var_name} = "
                                f"{self.get_gate_code(left_var, right_var, gate_op)};"
                            )

        return code

    def _get_pooling_layer_code(self, pool_info: Dict[str, Any], layer_name: str, input_shape: tuple) -> List[str]:
        """Generate C code for an OrPooling (max pooling) layer."""
        code = []

        if len(input_shape) == 3:
            pool_dim = 2
            channels, in_h, in_w = input_shape
        elif len(input_shape) == 4:
            pool_dim = 3
            channels, in_h, in_w, in_d = input_shape
        else:
            raise ValueError(f"Pool layer expects 3D or 4D input, got {len(input_shape)}D")

        kernel_size = pool_info['kernel_size']
        stride = pool_info['stride']
        padding = pool_info['padding']

        # Calculate output dimensions
        out_h = ((in_h + 2 * padding - kernel_size) // stride) + 1
        out_w = ((in_w + 2 * padding - kernel_size) // stride) + 1
        if pool_dim==3:
            out_d = ((in_d + 2 * padding - kernel_size) // stride) + 1

        code.append(f"\t// Max pooling layer {layer_name}")
        if pool_dim==2:
            code.append(f"\t// Input: {channels}x{in_h}x{in_w}, Output: {channels}x{out_h}x{out_w}")
        else:
            code.append(f"\t// Input: {channels}x{in_h}x{in_w}x{in_d}, Output: {channels}x{out_h}x{out_w}x{out_d}")

        # Generate pooling code for each channel and output position
        prev_layer_name = self._get_previous_layer_name(layer_name)

        for c in range(channels):
            if pool_dim==2:
                for out_y in range(out_h):
                    for out_x in range(out_w):
                        # Calculate input region for this output position
                        in_y_start = out_y * stride - padding
                        in_x_start = out_x * stride - padding
                        in_y_end = min(in_y_start + kernel_size, in_h)
                        in_x_end = min(in_x_start + kernel_size, in_w)
                        in_y_start = max(in_y_start, 0)
                        in_x_start = max(in_x_start, 0)

                        output_idx = c * out_h * out_w + out_y * out_w + out_x

                        # Initialize with first valid input
                        first_input_idx = c * in_h * in_w + in_y_start * in_w + in_x_start
                        if prev_layer_name == "inp":
                            code.append(f"\tlayer_{layer_name}_out[{output_idx}] = inp[{first_input_idx}];")
                        else:
                            code.append(f"\tlayer_{layer_name}_out[{output_idx}] = layer_{prev_layer_name}_out[{first_input_idx}];")

                        # Compare with remaining inputs in the kernel window (max pooling using OR)
                        for ky in range(in_y_start, in_y_end):
                            for kx in range(in_x_start, in_x_end):
                                if ky == in_y_start and kx == in_x_start:
                                    continue  # Skip first element (already initialized)

                                input_idx = c * in_h * in_w + ky * in_w + kx
                                if prev_layer_name == "inp":
                                    input_var = f"inp[{input_idx}]"
                                else:
                                    input_var = f"layer_{prev_layer_name}_out[{input_idx}]"

                                # Max operation using bitwise OR (since we're working with binary values)
                                code.append(f"\tlayer_{layer_name}_out[{output_idx}] |= {input_var};")
            else:
                for out_z in range(out_h):
                    for out_y in range(out_w):
                        for out_x in range(out_d):
                            # Calculate input region for this output position
                            in_z_start = out_z * stride - padding
                            in_y_start = out_y * stride - padding
                            in_x_start = out_x * stride - padding
                            in_z_end = min(in_z_start + kernel_size, in_h)
                            in_y_end = min(in_y_start + kernel_size, in_w)
                            in_x_end = min(in_x_start + kernel_size, in_d)
                            in_z_start = max(in_z_start, 0)
                            in_y_start = max(in_y_start, 0)
                            in_x_start = max(in_x_start, 0)

                            output_idx = (
                                c * out_h * out_w * out_d
                                + out_z * out_w * out_d
                                + out_y * out_d
                                + out_x
                             )

                            # Initialize with first valid input
                            first_input_idx = (
                                c * in_h * in_w * in_d
                                + in_z_start * in_w * in_d
                                + in_y_start * in_d
                                + in_x_start
                            )
                            if prev_layer_name == "inp":
                                code.append(f"\tlayer_{layer_name}_out[{output_idx}] = inp[{first_input_idx}];")
                            else:
                                code.append(f"\tlayer_{layer_name}_out[{output_idx}] = layer_{prev_layer_name}_out[{first_input_idx}];")

                            # Compare with remaining inputs in the kernel window (max pooling using OR)
                            for kz in range(in_z_start, in_z_end):
                                for ky in range(in_y_start, in_y_end):
                                    for kx in range(in_x_start, in_x_end):
                                        if (ky == in_y_start and kx == in_x_start) and kz == in_z_start:
                                            continue  # Skip first element (already initialized)

                                        input_idx = (
                                            c * in_h * in_w * in_d
                                            + kz * in_w * in_d
                                            + ky * in_d
                                            + kx
                                        )
                                        if prev_layer_name == "inp":
                                            input_var = f"inp[{input_idx}]"
                                        else:
                                            input_var = f"layer_{prev_layer_name}_out[{input_idx}]"

                                        # Max operation using bitwise OR (since we're working with binary values)
                                        code.append(f"\tlayer_{layer_name}_out[{output_idx}] |= {input_var};")

        return code

    def _get_previous_layer_name(self, current_layer_name: str) -> str:
        """Get the name of the previous layer in execution order."""
        layer_info = self._calculate_layer_output_sizes_and_shapes()

        # Find current layer in execution order
        current_idx = None
        for i, (layer_type, layer_idx, _, _) in enumerate(layer_info):
            if f"{layer_type}_{layer_idx}" == current_layer_name:
                current_idx = i
                break

        if current_idx is None or current_idx == 0:
            return "inp"  # First layer uses input

        # Get previous layer info
        prev_layer_type, prev_layer_idx, _, _ = layer_info[current_idx - 1]

        if prev_layer_type == 'flatten':
            # Flatten doesn't create output, so go one more layer back
            if current_idx >= 2:
                prev_prev_layer_type, prev_prev_layer_idx, _, _ = layer_info[current_idx - 2]
                return f"{prev_prev_layer_type}_{prev_prev_layer_idx}"
            else:
                return "inp"

        return f"{prev_layer_type}_{prev_layer_idx}"

    def _get_linear_layer_code(self, layer_a, layer_b, layer_op, layer_id: int, is_final: bool, has_non_linear_layers: bool) -> List[str]:
        """Generate C code for a linear layer."""
        code = []

        has_flatten = any(lt == 'flatten' for lt, _, _, _ in self._calculate_layer_output_sizes_and_shapes())

        # Determine input source based on layer position and model structure
        if has_non_linear_layers or has_flatten:
            # Mixed model or model with flatten: first layer uses linear_input, subsequent layers alternate
            if layer_id == 0:
                input_var = "linear_input"
            else:
                # Alternate between linear_input and linear_buf_temp
                if layer_id % 2 == 1:
                    input_var = "linear_buf_temp"
                else:
                    input_var = "linear_input"
        elif layer_id == 0:
            # First layer in linear-only model: use direct input
            input_var = "inp"
        else:
            # Subsequent layers in linear-only model: alternate between buffers
            if layer_id % 2 == 1:
                input_var = "linear_buf_a"
            else:
                input_var = "linear_buf_b"

        # Determine output destination
        if is_final:
            output_var = "out"
        elif has_non_linear_layers or has_flatten:
            # Mixed model or flatten model: alternate between buffers
            if layer_id % 2 == 0:
                output_var = "linear_buf_temp"
            else:
                output_var = "linear_input"
        elif layer_id < len(self.linear_layers) - 1:
            # Multi-layer linear model: alternate between buffers
            if layer_id % 2 == 0:
                output_var = "linear_buf_a"
            else:
                output_var = "linear_buf_b"
        else:
            output_var = "out"

        for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
            a = f"{input_var}[{gate_a}]"
            b = f"{input_var}[{gate_b}]"
            code.append(f"\t{output_var}[{var_id}] = {self.get_gate_code(a, b, gate_op)};")

        return code

    def get_c_code(self) -> str:
        """Generate the complete C code for the network."""
        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "#include <string.h>",
            "#include <stdio.h>",
        ]

        if self.use_bitpacking:
            dtype = BITS_TO_DTYPE[self.num_bits]
        else:
            dtype = "bool"

        # Calculate sizes and shapes for all layers
        layer_info = self._calculate_layer_output_sizes_and_shapes()
        logic_net_inp_size = self._get_input_size()

        if self.apply_thresholding:
            if self.thresholding_layer is None:
                raise ValueError("If the inputs are not boolean then the first layer must be a thresholding layer.")
            logic_net_inp_size *= self.thresholding_layer['num_thresholds']

        if self.use_bitpacking:
            code.extend([
                "",
                f"void logic_net("
                f"{dtype} const *inp, "
                f"{dtype} *out);",
                "",
                f"void logic_net("
                f"{dtype} const *inp, "
                f"{dtype} *out) {{",
            ])

        else: # processing one event at a time
            code.extend([
                ""
                f"void logic_net("
                f"bool const inp[{logic_net_inp_size}], "
                f"bool out[{self._get_output_size()}]);",
                "",
                f"void logic_net("
                f"bool const inp[{logic_net_inp_size}], "
                f"bool out[{self._get_output_size()}]) {{",
            ])

        # Allocate intermediate buffers for non-linear layers
        for layer_type, layer_idx, output_shape, output_size in layer_info:
            if layer_type in ['conv', 'pool']:
                code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} layer_{layer_type}_{layer_idx}_out[{output_size}];")

        # Allocate buffers for linear layers if needed
        linear_layer_count = len(self.linear_layers)
        has_non_linear_layers = any(lt in ['conv', 'pool'] for lt, _, _, _ in layer_info)
        has_flatten = any(lt == 'flatten' for lt, _, _, _ in layer_info)

        if linear_layer_count > 0:
            if has_non_linear_layers or has_flatten:
                # Mixed model or model with flatten: need buffer to transfer data to linear layers
                # Find the size after flattening or from conv/pool layers
                flatten_size = None
                for layer_type, layer_idx, output_shape, output_size in layer_info:
                    if layer_type == 'flatten':
                        flatten_size = output_size
                        break
                
                # If no explicit flatten but we have conv/pool, use the last conv/pool output size
                if flatten_size is None and has_non_linear_layers:
                    for layer_type, layer_idx, output_shape, output_size in reversed(layer_info):
                        if layer_type in ['conv', 'pool']:
                            flatten_size = output_size
                            break

                if flatten_size:
                    # For multi-layer linear networks, we need the buffer to be large enough
                    # for the largest intermediate layer size
                    max_linear_input_size = flatten_size
                    if linear_layer_count > 1:
                        # Check all linear layer input sizes
                        for i, (layer_a, layer_b, layer_op) in enumerate(self.linear_layers[:-1]):
                            layer_output_size = len(layer_a)  # This layer's output size
                            max_linear_input_size = max(max_linear_input_size, layer_output_size)

                    code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_input[{max_linear_input_size}];")

            # For multi-layer linear networks, use ping-pong buffers
            if linear_layer_count > 1:
                if has_non_linear_layers or has_flatten:
                    # Mixed model or flatten model: use one additional buffer for ping-ponging
                    max_linear_size = max(len(layer[0]) for layer in self.linear_layers)
                    code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_buf_temp[{max_linear_size}];")
                else:
                    # Linear-only model: use two ping-pong buffers
                    max_linear_size = max(len(layer[0]) for layer in self.linear_layers)
                    code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_buf_a[{max_linear_size}];")
                    code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_buf_b[{max_linear_size}];")

        # Check if we need a flatten buffer when no linear layers follow (for GroupSum only)
        if has_flatten and linear_layer_count == 0:
            # Need buffer for flatten → GroupSum case
            flatten_size = None
            for layer_type, layer_idx, output_shape, output_size in layer_info:
                if layer_type == 'flatten':
                    flatten_size = output_size
                    break
            if flatten_size:
                code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} flattened_output[{flatten_size}];")

        code.append("")

        # Generate code for all layers in execution order
        linear_layer_counter = 0
        current_shape = self.input_shape

        for layer_type, layer_idx, output_shape, output_size in layer_info:
            if layer_type == 'conv':
                layer_name = f"{layer_type}_{layer_idx}"
                code.extend(self._get_conv_layer_code(self.conv_layers[layer_idx], layer_name))
                code.append("")
            elif layer_type == 'pool':
                layer_name = f"{layer_type}_{layer_idx}"
                code.extend(self._get_pooling_layer_code(self.pooling_layers[layer_idx], layer_name, current_shape))
                code.append("")
            elif layer_type == 'flatten':
                # Handle flattening: copy from input or previous layer to appropriate buffer
                current_layer_idx = layer_info.index((layer_type, layer_idx, output_shape, output_size))
                
                if current_layer_idx == 0:
                    # First layer is flatten: copy directly from input
                    if linear_layer_count > 0:
                        code.append(f"\t// Flatten layer: copy from input to linear_input")
                        code.append(f"\tmemcpy(linear_input, inp, "
                                   f"{output_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
                    else:
                        code.append(f"\t// Flatten layer: copy from input to flattened_output")
                        code.append(f"\tmemcpy(flattened_output, inp, "
                                   f"{output_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
                else:
                    # Flatten follows other layers: copy from previous layer
                    prev_layer_info = layer_info[current_layer_idx - 1]
                    prev_layer_type, prev_layer_idx, prev_shape, prev_size = prev_layer_info

                    if linear_layer_count > 0:
                        # Flatten → Linear case: copy to linear_input
                        code.append(f"\t// Flatten layer: copy from layer_{prev_layer_type}_{prev_layer_idx}_out to linear_input")
                        code.append(f"\tmemcpy(linear_input, layer_{prev_layer_type}_{prev_layer_idx}_out, "
                                   f"{prev_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
                    else:
                        # Flatten → GroupSum case: copy to flattened_output buffer
                        code.append(f"\t// Flatten layer: copy from layer_{prev_layer_type}_{prev_layer_idx}_out to flattened_output")
                        code.append(f"\tmemcpy(flattened_output, layer_{prev_layer_type}_{prev_layer_idx}_out, "
                                   f"{prev_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
                code.append("")
            elif layer_type == 'linear':
                layer_a, layer_b, layer_op = self.linear_layers[layer_idx]
                is_final = (linear_layer_counter == linear_layer_count - 1)
                has_non_linear = any(lt in ['conv', 'pool'] for lt, _, _, _ in layer_info)
                code.extend(self._get_linear_layer_code(layer_a, layer_b, layer_op, linear_layer_counter, is_final, has_non_linear))
                linear_layer_counter += 1

            current_shape = output_shape

        # Handle case where we only have conv/pool layers (no linear layers)
        if linear_layer_count == 0 and layer_info:
            final_layer_type, final_layer_idx, _, final_size = layer_info[-1]
            if final_layer_type in ['conv', 'pool']:
                code.append(f"\t// Copy final layer output to result")
                code.append(f"\tmemcpy(out, layer_{final_layer_type}_{final_layer_idx}_out, "
                           f"{final_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
            elif final_layer_type == 'flatten':
                # Flatten is the final operation, copy from flattened_output to out
                code.append(f"\t// Copy flattened output to result")
                code.append(f"\tmemcpy(out, flattened_output, "
                           f"{final_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")

        code.append("}")

        # Add processing function
        if self.use_bitpacking:
            code.extend(self._generate_batch_processing_function())
        else:
            code.extend(self._generate_single_processing_function())

        return "\n".join(code)

    def _generate_batch_processing_function(self) -> List[str]:
        """Generate the batch processing function for GroupSum."""
        input_size = self._get_input_size()
        output_size = self._get_output_size()

        num_neurons_ll = output_size
        log2_of_num_neurons_per_class_ll = math.ceil(
            math.log2(num_neurons_ll / self.num_classes + 1)
        )

        return [
            f"""
void apply_logic_net(bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc({input_size}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc({num_neurons_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp_o = malloc({log2_of_num_neurons_per_class_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));

    for(size_t i = 0; i < len; ++i) {{

        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < {input_size}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(inp[i * {input_size} * {self.num_bits} + ({self.num_bits} - b - 1) * {input_size} + d]);
            }}
            inp_temp[d] = res;
        }}

        // Applying the logic net
        logic_net(inp_temp, out_temp);

        // GroupSum of the results via logic gate networks
        for(size_t c = 0; c < {self.num_classes}; ++c) {{  // for each class
            // Initialize the output bits
            for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                out_temp_o[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            }}

            // Apply the adder logic gate network
            for(size_t a = 0; a < {num_neurons_ll // self.num_classes}; ++a) {{
                {BITS_TO_DTYPE[self.num_bits]} carry = out_temp[c * {num_neurons_ll // self.num_classes} + a];
                {BITS_TO_DTYPE[self.num_bits]} out_temp_o_d;
                for(int d = {log2_of_num_neurons_per_class_ll} - 1; d >= 0; --d) {{
                    out_temp_o_d  = out_temp_o[d];
                    out_temp_o[d] = carry ^ out_temp_o_d;
                    carry         = carry & out_temp_o_d;
                }}
            }}

            // Unpack the result bits
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                const {BITS_TO_DTYPE[self.num_bits]} bit_mask = {BITS_TO_ONE_LITERAL[self.num_bits]} << b;
                {BITS_TO_DTYPE[32]} res = 0;
                for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                    res <<= 1;
                    res += !!(out_temp_o[d] & bit_mask);
                }}
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = (res + {self.beta}) / {self.tau};
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}
"""
        ]

    def _generate_single_processing_function(self) -> List[str]:
        """Generate the processing function for the case where we want to process one event at a time.
        Includes thresholding if apply_thresholding is set to True. Also includes GroupSum. 
        This method is meant to be the top-level method for HLS synthesis."""

        input_size = self._get_input_size()
        num_neurons_ll = self._get_output_size()

        if self.apply_thresholding:
            thresholding_info = self.thresholding_layer
            num_thresholds = thresholding_info["num_thresholds"]
            thresholds = thresholding_info["thresholds"].tolist()
            thresholds_str = ", ".join(str(int(x)) for x in thresholds)
            thresholds_c_def = f"const int thresholds[{len(thresholds)}] = {{{thresholds_str}}};"
        
        inp_dtype = BITS_TO_DTYPE[32] if self.apply_thresholding else "bool"
        out_len = self.num_classes
        logic_net_out_arg = "out_temp"
        out_dtype = BITS_TO_DTYPE[32]

        code = [
            f"""void apply_logic_net_one_event({inp_dtype} const inp[{input_size}], {BITS_TO_DTYPE[32]} out[{out_len}]) {{
    bool out_temp[{num_neurons_ll}];"""
        ]

        if self.apply_thresholding:
            code.append(f"""    bool inp_temp[{input_size * num_thresholds}];
    {thresholds_c_def}

    // Convert the inputs into boolean
    for (size_t t = 0; t < {num_thresholds}; t++) {{
        for (size_t in_idx = 0; in_idx < {input_size}; in_idx++){{
            size_t out_idx = t * {input_size} + in_idx;
            inp_temp[out_idx] = (inp[in_idx] > thresholds[t])? 1 : 0;
        }}
    }}
    
    //run logic net
    logic_net(inp_temp, {logic_net_out_arg});
""")
        else:
            code.append(f"""
    //run logic net
    logic_net(inp, {logic_net_out_arg});
""")

        code.append(f"""
    //apply GroupSum
    for (size_t c = 0; c < {self.num_classes}; c++) {{
        {BITS_TO_DTYPE[32]} sum = 0;
        for(size_t a = 0; a < {num_neurons_ll} / {self.num_classes}; a++) {{
            sum += out_temp[c * {num_neurons_ll} / {self.num_classes} + a];
        }}""")

        if (self.beta!=0) or (self.tau!=1):
                code.append(f"""
        out[c] = (sum + {self.beta}) / {self.tau};
    }}
}}
""")
        else:
                code.append(f"""
        out[c] = sum;
    }}
}}
""")

        code.append(
            f"""
    
void apply_logic_net({inp_dtype} const *inp, {out_dtype} *out, size_t len) {{
    {inp_dtype} *inp_temp = malloc({input_size}*sizeof({inp_dtype}));
    {out_dtype} *out_temp = malloc({out_len}*sizeof({out_dtype}));

    // run inference one event at a time
    for (size_t i = 0; i < len; ++i) {{
        for (size_t j = 0; j < {input_size}; ++j) {{
            inp_temp[j] = inp[i * {input_size} + j];
        }}
        apply_logic_net_one_event(inp_temp, out_temp);
        for (size_t k = 0; k < {out_len}; ++k) {{
            out[i * {out_len} + k] = out_temp[k];
        }}
    }}
    free(inp_temp);
    free(out_temp);
}}

""")
        return code
    

    def _get_input_size(self) -> int:
        """Get the total input size."""
        if self.input_shape is None:
            raise ValueError("Input shape not set")
        return np.prod(self.input_shape)

    def _get_output_size(self) -> int:
        """Get the total output size."""
        if self.linear_layers:
            return len(self.linear_layers[-1][0])
        else:
            # Get the final layer info
            layer_info = self._calculate_layer_output_sizes_and_shapes()
            if layer_info:
                return layer_info[-1][3]  # output_size
            else:
                raise ValueError("No layers found")

    def compile(self, opt_level: int = 1, save_lib_path: str = None, verbose: bool = False):
        """Compile the network to a shared library."""
        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c") as c_file:
                code = self.get_c_code()

                if verbose and len(code.split("\n")) <= 200:
                    print("\n" + code + "\n")

                c_file.write(code)
                c_file.flush()

                if verbose:
                    n_lines = len(code.split('\n'))
                    print(f"C code created with {n_lines} lines. (temp location {c_file.name})")

                t_s = time.time()
                compiler_out = subprocess.run([
                    self.cpu_compiler, "-shared", "-fPIC", f"-O{opt_level}",
                    "-o", lib_file.name, c_file.name,
                ])

                if compiler_out.returncode != 0:
                    raise RuntimeError(f"compilation exited with error code {compiler_out.returncode}")

                print(f"Compiling finished in {time.time() - t_s:.3f} seconds.")

            if save_lib_path is not None:
                shutil.copy(lib_file.name, save_lib_path)
                if verbose:
                    print(f"lib_file copied from {lib_file.name} to {save_lib_path}")

            lib = ctypes.cdll.LoadLibrary(lib_file.name)
            self._setup_library_function(lib)

    def _setup_library_function(self, lib):
        """Setup the library function with appropriate signatures."""
        inp_type = BITS_TO_C_DTYPE[32] if self.apply_thresholding else ctypes.c_bool
        out_type = BITS_TO_C_DTYPE[32]

        lib_fn = lib.apply_logic_net
        lib_fn.restype = None
        lib_fn.argtypes = [
            np.ctypeslib.ndpointer(inp_type, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(out_type, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        self.lib_fn = lib_fn

    def forward(
        self,
        x: Union[torch.BoolTensor, numpy.typing.NDArray[np.bool_]],
        verbose: bool = False,
    ) -> torch.IntTensor:
        """Forward pass through the compiled network."""
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        return self._forward(x, verbose)

    def _forward(self, x: np.ndarray, verbose: bool) -> torch.IntTensor:
        """Forward pass."""
        if self.use_bitpacking:
            batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
            pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
            x = np.concatenate([x, np.zeros_like(x[:pad_len])])
            n_iter = batch_size_div_bits
        else:
            pad_len = 0
            n_iter = x.shape[0]

        if verbose:
            print("x.shape", x.shape)

        out = np.zeros(x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32])
        x = x.reshape(-1)

        self.lib_fn(x, out, n_iter)

        if self.use_bitpacking:
            out = torch.tensor(out).view(batch_size_div_bits * self.num_bits, self.num_classes)
        else:
            out = torch.tensor(out).view(n_iter, self.num_classes)

        if pad_len > 0:
            out = out[:-pad_len]

        if verbose:
            print("out.shape", out.shape)

        return out

    @staticmethod
    def load(save_lib_path: str, input_shape: tuple, num_classes: int = None, num_bits: int = 64):
        """Load a compiled network from a shared library.

        Note: input_shape is required here since we're loading a pre-compiled library
        without access to the original model structure for inference.
        """
        self = CompiledLogicNet(None, num_bits=num_bits)
        self.input_shape = input_shape
        self.num_classes = num_classes

        lib = ctypes.cdll.LoadLibrary(save_lib_path)
        self._setup_library_function(lib)
        return self
