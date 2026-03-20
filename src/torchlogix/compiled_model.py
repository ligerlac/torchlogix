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

from .layers.conv import LogicConv2d, LogicConv3d
from .layers.pool import OrPooling2d, OrPooling3d
from .layers.dense import LogicDense
from .layers.groupsum import GroupSum
from .layers.binarization import Binarization


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
        apply_groupsum_scaling: bool = True,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.apply_thresholding = False  # Will be set during _parse_model
        self.use_bitpacking = use_bitpacking  # Store user's choice
        self.cpu_compiler = cpu_compiler
        self.apply_groupsum_scaling = apply_groupsum_scaling

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

        # Detect if first layer is thresholding (but not DummyBinarization)
        if len(self.model) > 0 and isinstance(self.model[0], Binarization):
            # Check if it's actually a real thresholding layer (not dummy)
            if self.model[0].thresholds is not None:
                self.apply_thresholding = True

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
            if isinstance(layer, Binarization):
                if len(self.layer_order) > 0:
                    raise ValueError("Binarization layer must appear first in layer order.")
                # Only extract thresholding info for real thresholding layers (not dummy)
                if layer.thresholds is not None:
                    thresholding_info = self._extract_thresholding_layer_info(layer)
                    self.thresholding_layer = thresholding_info  # there will only be one thresholding layer
            elif isinstance(layer, (LogicConv2d, LogicConv3d)):
                conv_info = self._extract_conv_layer_info(layer)
                self.conv_layers.append(conv_info)
                self.layer_order.append(('conv', len(self.conv_layers) - 1))
            elif isinstance(layer, (OrPooling2d, OrPooling3d)):
                pool_info = self._extract_pooling_layer_info(layer)
                self.pooling_layers.append(pool_info)
                self.layer_order.append(('pool', len(self.pooling_layers) - 1))
            elif isinstance(layer, LogicDense):
                self.linear_layers.append(
                    (layer.connections.indices[0], layer.connections.indices[1], layer.weight.argmax(1))
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

    def _extract_thresholding_layer_info(self, layer: Binarization) -> Dict[str, Any]:
        """Extract information from a thresholding layer for compilation."""
        thresholds = layer.get_thresholds()
        # Detect if thresholds are float or int
        is_float = thresholds.dtype.is_floating_point
        return {
            # 'num_thresholds': layer.num_thresholds,
            'num_thresholds': len(layer.thresholds),
            'thresholds': layer.get_thresholds(),
            'is_float': is_float,
        }

    def _extract_conv_layer_info(self, layer: Union[LogicConv2d, LogicConv3d]) -> Dict[str, Any]:
        """Extract information from a LogicConv2d or LogicConv3d layer for compilation."""
        tree_operations = []
        for level_idx, level_weights in enumerate(layer.tree_weights):
            # level_weights shape: (c, f, lut_entries) where c=num_kernels, f=num_features
            # We need structure [level][feature][kernel]
            # Transpose to (f, c, lut_entries) then extract ops
            level_weights_transposed = level_weights.transpose(0, 1)
            level_ops = []
            for weight_param in level_weights_transposed:
                ops = weight_param.argmax(1).cpu().numpy()
                level_ops.append(ops)
            tree_operations.append(level_ops)

        return {
            'indices': layer.connections.indices,
            'tree_operations': tree_operations,
            'tree_depth': layer.tree_depth,
            'num_kernels': layer.num_kernels,
            'in_dim': layer.in_dim,
            'receptive_field_size': layer.receptive_field_size,
            'stride': layer.stride,
            'padding': layer.padding,
            'channels': layer.channels,
        }

    def _extract_pooling_layer_info(self, layer: Union[OrPooling2d, OrPooling3d]) -> Dict[str, Any]:
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

    def get_gate_verilog(self, var1: str, var2: str, gate_op: int) -> str:
        """Generate Verilog code for a logic gate operation.

        Args:
            var1: Name of first input variable (Verilog syntax)
            var2: Name of second input variable (Verilog syntax)
            gate_op: Gate operation ID (0-15)

        Returns:
            Verilog expression string
        """
        from .hdl_generator import gate_id_to_verilog
        print(f"gate_op = {gate_op}, var1={var1}, var2={var2}")
        print(f"type of gate_op: {type(gate_op)}")
        return gate_id_to_verilog(gate_op, var1, var2)

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
                    h_out = ((h + 2 * conv_info['padding'] - conv_info['receptive_field_size'][0])
                             // conv_info['stride']) + 1
                    w_out = ((w + 2 * conv_info['padding'] - conv_info['receptive_field_size'][1])
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
    
    # Helper function to check if coordinates are in padding region
    def _is_padded(self, coords, conv_dim, in_dim, padding):
        """Check if any coordinate is in the padding region."""
        if padding == 0:
            return False
        
        if conv_dim == 2:
            h, w = coords[0], coords[1]
            return (h < padding or h >= in_dim[0] + padding or
                    w < padding or w >= in_dim[1] + padding)
        else:  # conv_dim == 3
            h, w, d = coords[0], coords[1], coords[2]
            return (h < padding or h >= in_dim[0] + padding or
                    w < padding or w >= in_dim[1] + padding or
                    d < padding or d >= in_dim[2] + padding)
        
    # Helper function to generate input variable reference
    def _get_input_var(self, coords, channel, source_name, conv_dim, in_dim, padding):
        """Generate input variable reference with padding handling.
        
        Args:
            coords: tuple of (h, w) for 2D or (h, w, d) for 3D
            channel: channel index
            source_name: "inp" or "layer_{prev_layer_name}_out"
        
        Returns:
            String representing the input variable (e.g., "inp[idx]" or "0")
        """
        # Check if in padding region
        if self._is_padded(coords, conv_dim, in_dim, padding):
            return "0"  # Padded values are zero
        
        # Adjust coordinates if padding is used
        if padding > 0:
            if conv_dim == 2:
                h, w = coords
                h_adj = h - padding
                w_adj = w - padding
                flat_idx = (f"{channel} * {in_dim[0]} * {in_dim[1]} + "
                           f"{h_adj} * {in_dim[1]} + {w_adj}")
            else:  # conv_dim == 3
                h, w, d = coords
                h_adj = h - padding
                w_adj = w - padding
                d_adj = d - padding
                flat_idx = (f"{channel} * {in_dim[0]} * {in_dim[1]} * {in_dim[2]} + "
                           f"{h_adj} * {in_dim[1]} * {in_dim[2]} + "
                           f"{w_adj} * {in_dim[2]} + {d_adj}")
        else:
            # No padding, use coordinates directly
            if conv_dim == 2:
                h, w = coords
                flat_idx = (f"{channel} * {in_dim[0]} * {in_dim[1]} + "
                           f"{h} * {in_dim[1]} + {w}")
            else:  # conv_dim == 3
                h, w, d = coords
                flat_idx = (f"{channel} * {in_dim[0]} * {in_dim[1]} * {in_dim[2]} + "
                           f"{h} * {in_dim[1]} * {in_dim[2]} + "
                           f"{w} * {in_dim[2]} + {d}")
        
        return f"{source_name}[{flat_idx}]"

    def _get_flat_index(self, coords, channel, conv_dim, in_dim, padding):
        """Compute flat index into input array with padding handling.
        
        Args:
            coords: tuple of (h, w) for 2D or (h, w, d) for 3D
            channel: channel index
        
        Returns:
            Integer flat index, or None if in padding region
        """

        if self._is_padded(coords, conv_dim, in_dim, padding):
            return None  # Will be handled as constant 0
        
        # Adjust coordinates if padding is used
        if padding > 0:
            if conv_dim == 2:
                h, w = coords
                h_adj = h - padding
                w_adj = w - padding
                return channel * in_dim[0] * in_dim[1] + h_adj * in_dim[1] + w_adj
            else:  # conv_dim == 3
                h, w, d = coords
                h_adj = h - padding
                w_adj = w - padding
                d_adj = d - padding
                return (channel * in_dim[0] * in_dim[1] * in_dim[2] +
                       h_adj * in_dim[1] * in_dim[2] +
                       w_adj * in_dim[2] + d_adj)
        else:
            # No padding, use coordinates directly
            if conv_dim == 2:
                h, w = coords
                return channel * in_dim[0] * in_dim[1] + h * in_dim[1] + w
            else:  # conv_dim == 3
                h, w, d = coords
                return (channel * in_dim[0] * in_dim[1] * in_dim[2] +
                       h * in_dim[1] * in_dim[2] +
                       w * in_dim[2] + d)

    def _get_conv_layer_code(self, conv_info: Dict[str, Any], layer_name: str) -> List[str]:
        """Generate C code for a convolutional layer."""
        code = []
        indices = conv_info['indices']
        tree_ops = conv_info['tree_operations']
        padding = conv_info['padding']
        in_dim = conv_info['in_dim']

        if len(conv_info['in_dim']) == 2:
            conv_dim = 2
            h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][0])
                    // conv_info['stride']) + 1
            w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][1])
                    // conv_info['stride']) + 1
            iter_range = h_out * w_out
        elif len(conv_info['in_dim']) == 3:
            conv_dim = 3
            h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][0])
                    // conv_info['stride']) + 1
            w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][1])
                    // conv_info['stride']) + 1
            d_out = ((conv_info['in_dim'][2] + 2 * conv_info['padding'] - conv_info['receptive_field_size'][2])
                    // conv_info['stride']) + 1
            iter_range = h_out * w_out * d_out
        else:
            raise ValueError(f"Conv layer expects 3D or 4D input, got {len(conv_info['in_dim'])}")

        code.append(f"\t// Convolutional layer {layer_name}")

        for kernel_idx in range(conv_info['num_kernels']):
            for pos_idx in range(iter_range):
                # First level: process receptive field positions
                level_0_indices = indices[0]
                # Note: indices structure is (L, P, K, S, 3) so we swap kernel_idx and pos_idx
                left_indices = level_0_indices[0][pos_idx, kernel_idx]
                right_indices = level_0_indices[1][pos_idx, kernel_idx]

                # Generate variables for the first level
                for gate_idx in range(2**(conv_info['tree_depth']-1)):
                    if conv_dim == 2:
                        left_h, left_w, left_c = left_indices[gate_idx]
                        right_h, right_w, right_c = right_indices[gate_idx]
                        left_coords = (left_h, left_w)
                        right_coords = (right_h, right_w)
                    else:
                        left_h, left_w, left_d, left_c = left_indices[gate_idx]
                        right_h, right_w, right_d, right_c = right_indices[gate_idx]
                        left_coords = (left_h, left_w, left_d)
                        right_coords = (right_h, right_w, right_d)

                    gate_op = tree_ops[0][gate_idx][kernel_idx]

                    # Determine input source
                    prev_layer_name = self._get_previous_layer_name(layer_name)
                    if prev_layer_name == "inp":
                        source_name = "inp"
                    else:
                        source_name = f"layer_{prev_layer_name}_out"

                    # Generate input variable references with padding handling
                    left_var = self._get_input_var(left_coords, left_c, source_name, conv_dim, in_dim, padding)
                    right_var = self._get_input_var(right_coords, right_c, source_name, conv_dim, in_dim, padding)

                    var_name = f"conv_{layer_name}_k{kernel_idx}_p{pos_idx}_l0_g{gate_idx}"
                    code.append(
                        f"\tconst {BITS_TO_DTYPE[self.num_bits]} {var_name} = "
                        f"{self.get_gate_code(left_var, right_var, gate_op)};"
                    )

                # Process remaining tree levels
                for level in range(1, conv_info['tree_depth']):
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

                        if level == conv_info['tree_depth'] - 1:
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

    def _get_linear_layer_verilog(self, layer_a, layer_b, layer_op,
                                   layer_id: int, input_name: str,
                                   output_name: str) -> List[str]:
        """Generate Verilog code for a linear (LogicDense) layer.

        Args:
            layer_a: Indices of first inputs for each neuron
            layer_b: Indices of second inputs for each neuron
            layer_op: Gate operations for each neuron (0-15)
            layer_id: Layer index
            input_name: Name of input wire/bus
            output_name: Name of output wire/bus

        Returns:
            List of Verilog code lines (wire declarations and assignments)
        """
        from .hdl_generator import format_verilog_comment

        code = []
        code.append("")
        code.append(format_verilog_comment(f"Layer {layer_id}: LogicDense ({len(layer_op)} neurons)"))

        for neuron_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
            # Input signal names
            a_signal = f"{input_name}[{gate_a}]"
            b_signal = f"{input_name}[{gate_b}]"

            # Generate gate expression
            gate_expr = self.get_gate_verilog(a_signal, b_signal, gate_op)

            # Create assignment
            code.append(f"    assign {output_name}[{neuron_id}] = {gate_expr};")

        return code

    def _get_conv_layer_verilog(self, conv_info: Dict[str, Any], layer_id: int,
                                 input_name: str, output_name: str) -> List[str]:
        """Generate Verilog code for a convolutional (LogicConv2d/3d) layer.

        Args:
            conv_info: Dictionary containing convolutional layer information
            layer_id: Layer index
            input_name: Name of input wire/bus
            output_name: Name of output wire/bus

        Returns:
            List of Verilog code lines (wire declarations and assignments)
        """
        from .hdl_generator import format_verilog_comment

        code = []
        code.append("")
        code.append(format_verilog_comment(f"Layer {layer_id}: LogicConv (tree_depth={conv_info['tree_depth']})"))

        indices = conv_info['indices']
        tree_ops = conv_info['tree_operations']
        padding = conv_info['padding']
        in_dim = conv_info['in_dim']

        conv_dim = len(in_dim)
        assert conv_dim in [2, 3], f"Expected 2D or 3D conv, got {conv_dim}D"

        # Calculate output dimensions
        if conv_dim == 2:
            print(f"conv_info['in_dim'] = {in_dim}")
            print(f"conv_info['padding'] = {padding}")
            print(f"conv_info['receptive_field_size'] = {conv_info['receptive_field_size']}")
            h_out = ((in_dim[0] + 2 * padding - conv_info['receptive_field_size'][0])
                    // conv_info['stride']) + 1
            w_out = ((in_dim[1] + 2 * padding - conv_info['receptive_field_size'][1])
                    // conv_info['stride']) + 1
            iter_range = h_out * w_out
        else:  # conv_dim == 3
            h_out = ((in_dim[0] + 2 * padding - conv_info['receptive_field_size'][0])
                    // conv_info['stride']) + 1
            w_out = ((in_dim[1] + 2 * padding - conv_info['receptive_field_size'][1])
                    // conv_info['stride']) + 1
            d_out = ((in_dim[2] + 2 * padding - conv_info['receptive_field_size'][2])
                    // conv_info['stride']) + 1
            iter_range = h_out * w_out * d_out

        # Generate wire declarations for intermediate tree levels
        for kernel_idx in range(conv_info['num_kernels']):
            for pos_idx in range(iter_range):
                # Wires for all tree levels (except final output)
                for level in range(conv_info['tree_depth']):
                    num_gates = 2 ** (conv_info['tree_depth'] - level)
                    for gate_idx in range(num_gates):
                        wire_name = f"conv_l{layer_id}_k{kernel_idx}_p{pos_idx}_lv{level}_g{gate_idx}"
                        code.append(f"    wire {wire_name};")

        code.append("")

        # Generate assignments for each kernel and position
        for kernel_idx in range(conv_info['num_kernels']):
            for pos_idx in range(iter_range):
                code.append(format_verilog_comment(f"Kernel {kernel_idx}, Position {pos_idx}"))

                # Level 0: leaf gates (process receptive field)
                level_0_indices = indices[0]
                # Note: indices structure is (L, P, K, S, 3) so we swap kernel_idx and pos_idx
                left_indices = level_0_indices[0][pos_idx, kernel_idx]
                right_indices = level_0_indices[1][pos_idx, kernel_idx]

                for gate_idx in range(2**(conv_info['tree_depth']-1)):
                    if conv_dim == 2:
                        left_h, left_w, left_c = left_indices[gate_idx]
                        right_h, right_w, right_c = right_indices[gate_idx]
                        left_coords = (left_h, left_w)
                        right_coords = (right_h, right_w)
                    else:  # conv_dim == 3
                        left_h, left_w, left_d, left_c = left_indices[gate_idx]
                        right_h, right_w, right_d, right_c = right_indices[gate_idx]
                        left_coords = (left_h, left_w, left_d)
                        right_coords = (right_h, right_w, right_d)

                    gate_op = tree_ops[0][gate_idx][kernel_idx]

                    # Get input signals with padding handling
                    left_flat_idx = self._get_flat_index(left_coords, left_c, conv_dim, in_dim, padding)
                    right_flat_idx = self._get_flat_index(right_coords, right_c, conv_dim, in_dim, padding)

                    # Generate input signal references
                    if left_flat_idx is None:
                        left_var = "1'b0"  # Padded region is zero
                    else:
                        left_var = f"{input_name}[{left_flat_idx}]"
                    if right_flat_idx is None:
                        right_var = "1'b0"  # Padded region is zero
                    else:
                        right_var = f"{input_name}[{right_flat_idx}]"

                    wire_name = f"conv_l{layer_id}_k{kernel_idx}_p{pos_idx}_lv0_g{gate_idx}"
                    gate_expr = self.get_gate_verilog(left_var, right_var, gate_op)
                    code.append(f"    assign {wire_name} = {gate_expr};")

                # Process remaining tree levels
                for level in range(1, conv_info['tree_depth']):
                    level_indices = indices[level]
                    left_gate_indices = level_indices[0]
                    right_gate_indices = level_indices[1]

                    num_gates = len(left_gate_indices)
                    for gate_idx in range(num_gates):
                        left_idx = left_gate_indices[gate_idx]
                        right_idx = right_gate_indices[gate_idx]

                        gate_op = tree_ops[level][gate_idx][kernel_idx]

                        left_var = f"conv_l{layer_id}_k{kernel_idx}_p{pos_idx}_lv{level-1}_g{left_idx}"
                        right_var = f"conv_l{layer_id}_k{kernel_idx}_p{pos_idx}_lv{level-1}_g{right_idx}"

                        gate_expr = self.get_gate_verilog(left_var, right_var, gate_op)

                        if level == conv_info['tree_depth']:
                            # Final level: assign to output
                            if conv_dim == 2:
                                output_idx = kernel_idx * h_out * w_out + pos_idx
                            else:
                                output_idx = kernel_idx * h_out * w_out * d_out + pos_idx
                            code.append(f"    assign {output_name}[{output_idx}] = {gate_expr};")
                        else:
                            # Intermediate level: assign to wire
                            wire_name = f"conv_l{layer_id}_k{kernel_idx}_p{pos_idx}_lv{level}_g{gate_idx}"
                            code.append(f"    assign {wire_name} = {gate_expr};")

                code.append("")

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

        # Thresholding setup
        if self.apply_thresholding:
            thresholding_info = self.thresholding_layer
            num_thresholds = thresholding_info["num_thresholds"]
            thresholds = thresholding_info["thresholds"].tolist()
            is_float = thresholding_info["is_float"]

            if is_float:
                thresholds_str = ", ".join(str(float(x)) for x in thresholds)
                thresholds_c_def = f"const float thresholds[{len(thresholds)}] = {{{thresholds_str}}};"
                inp_dtype = "float"
            else:
                thresholds_str = ", ".join(str(int(x)) for x in thresholds)
                thresholds_c_def = f"const int thresholds[{len(thresholds)}] = {{{thresholds_str}}};"
                inp_dtype = BITS_TO_DTYPE[32]

            logic_net_inp_size = input_size * num_thresholds
        else:
            inp_dtype = "bool"
            logic_net_inp_size = input_size

        # Output type: float if scaling is applied, int otherwise
        out_dtype = "float" if self.apply_groupsum_scaling else BITS_TO_DTYPE[32]

        code_start = f"""
void apply_logic_net({inp_dtype} const *inp, {out_dtype} *out, size_t len) {{
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc({logic_net_inp_size}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc({num_neurons_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp_o = malloc({log2_of_num_neurons_per_class_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));"""

        if self.apply_thresholding:
            code_start += f"""
    bool *bool_temp = malloc({logic_net_inp_size} * {self.num_bits} * sizeof(bool));
    {thresholds_c_def}"""

        code_start += f"""

    for(size_t i = 0; i < len; ++i) {{"""

        if self.apply_thresholding:
            code_threshold = f"""

        // Apply thresholding to convert inputs to boolean
        for (size_t b = 0; b < {self.num_bits}; ++b) {{
            for (size_t t = 0; t < {num_thresholds}; ++t) {{
                for (size_t in_idx = 0; in_idx < {input_size}; ++in_idx) {{
                    size_t out_idx = b * {logic_net_inp_size} + t * {input_size} + in_idx;
                    size_t inp_idx = i * {input_size} * {self.num_bits} + b * {input_size} + in_idx;
                    bool_temp[out_idx] = (inp[inp_idx] > thresholds[t]) ? 1 : 0;
                }}
            }}
        }}

        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < {logic_net_inp_size}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(bool_temp[b * {logic_net_inp_size} + d]);
            }}
            inp_temp[d] = res;
        }}"""
        else:
            code_threshold = f"""

        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < {input_size}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(inp[i * {input_size} * {self.num_bits} + ({self.num_bits} - b - 1) * {input_size} + d]);
            }}
            inp_temp[d] = res;
        }}"""

        return [
            code_start + code_threshold + f"""

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
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = {"(res + " + str(self.beta) + ") / " + str(self.tau) if self.apply_groupsum_scaling else "res"};
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);"""
            + (f"""
    free(bool_temp);""" if self.apply_thresholding else "") + """
}
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
            is_float = thresholding_info["is_float"]

            if is_float:
                thresholds_str = ", ".join(str(float(x)) for x in thresholds)
                thresholds_c_def = f"const float thresholds[{len(thresholds)}] = {{{thresholds_str}}};"
                inp_dtype = "float"
            else:
                thresholds_str = ", ".join(str(int(x)) for x in thresholds)
                thresholds_c_def = f"const int thresholds[{len(thresholds)}] = {{{thresholds_str}}};"
                inp_dtype = BITS_TO_DTYPE[32]
        else:
            inp_dtype = "bool"
        out_len = self.num_classes
        logic_net_out_arg = "out_temp"
        # Output type: float if scaling is applied, int otherwise
        out_dtype = "float" if self.apply_groupsum_scaling else BITS_TO_DTYPE[32]

        code = [
            f"""void apply_logic_net_one_event({inp_dtype} const inp[{input_size}], {out_dtype} out[{out_len}]) {{
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

        if self.apply_groupsum_scaling:
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

    def get_verilog_code(self, module_name: str = "torchlogix_net",
                        pipeline_stages: int = 0) -> str:
        """Generate complete Verilog code for the network.

        Args:
            module_name: Name of the top-level Verilog module
            pipeline_stages: Number of pipeline stages to insert (0 = fully combinational)
                           - 0: Fully combinational (no registers, 1 cycle latency, may not synthesize for large models)
                           - 1: Single output register (1 cycle latency, helps synthesis)
                           - N: Divide layers into N pipeline stages (N cycle latency)
                           - Use len(layers) for full layer-level pipelining

        Returns:
            Complete Verilog code as a string with specified pipelining

        Examples:
            # Fully combinational (original behavior)
            verilog = model.get_verilog_code(pipeline_stages=0)

            # Output register only (helps with large designs)
            verilog = model.get_verilog_code(pipeline_stages=1)

            # 4 pipeline stages (divide layers into 4 groups)
            verilog = model.get_verilog_code(pipeline_stages=4)

            # Full layer-level pipelining (register between each layer)
            verilog = model.get_verilog_code(pipeline_stages=999)  # or len(layers)
        """
        from .hdl_generator import (generate_verilog_module, format_verilog_comment,
                                    generate_pipeline_register)
        import numpy as np

        # Calculate layer sizes
        layer_info = self._calculate_layer_output_sizes_and_shapes()
        num_layers = len(layer_info)

        # Calculate input size from input_shape
        if self.input_shape is not None:
            input_size = int(np.prod(self.input_shape))
        else:
            # Fallback: get from first layer's input
            if len(self.linear_layers) > 0:
                input_size = len(self.linear_layers[0][0])  # First linear layer input size
            else:
                input_size = 1

        # Calculate output size from final layer
        # layer_info tuple is (layer_type, layer_idx, output_shape, output_size)
        if len(layer_info) > 0:
            output_size = layer_info[-1][3]  # Last layer's output size (4th element)
        else:
            output_size = 1

        # Determine pipelining strategy
        add_clock = pipeline_stages > 0
        pipeline_positions = []

        if pipeline_stages == 1:
            # Single output register only
            pipeline_positions = [num_layers]  # After all layers
        elif pipeline_stages > 1:
            # Distribute pipeline stages evenly across layers
            if pipeline_stages >= num_layers:
                # Register after every layer
                pipeline_positions = list(range(1, num_layers + 1))
            else:
                # Insert N-1 registers to create N stages
                # Distribute evenly across layers
                step = num_layers / pipeline_stages
                pipeline_positions = [int((i + 1) * step) for i in range(pipeline_stages - 1)]
                pipeline_positions.append(num_layers)  # Always register the output

        body_lines = []
        body_lines.append(format_verilog_comment("Generated by TorchLogix"))
        body_lines.append(format_verilog_comment(f"Input size: {input_size} bits"))
        body_lines.append(format_verilog_comment(f"Output size: {output_size} bits"))
        body_lines.append(format_verilog_comment(f"Pipeline stages: {len(pipeline_positions)}"))
        body_lines.append(format_verilog_comment(f"Latency: {len(pipeline_positions)} clock cycles"))
        body_lines.append("")

        # Track current input/output names for each layer
        current_input = "inp"
        layer_outputs = []
        wire_declarations = []
        register_code = []

        # Generate wire/reg declarations for layer outputs
        for idx, (layer_type, layer_idx, output_shape, output_size) in enumerate(layer_info):
            layer_num = idx + 1  # Layer numbers are 1-indexed

            if idx < len(layer_info) - 1:  # Not the final layer
                # Check if this layer should have a pipeline register
                if layer_num in pipeline_positions:
                    # This layer output needs a register
                    wire_name = f"layer_{idx}_comb"  # Combinational output
                    reg_name = f"layer_{idx}_out"   # Registered output
                    wire_declarations.append(f"    wire [{output_size-1}:0] {wire_name};")
                    register_code.append(generate_pipeline_register(reg_name, output_size, wire_name))
                    layer_outputs.append((wire_name, reg_name, output_size, True))  # Has register
                else:
                    # Just a wire
                    wire_name = f"layer_{idx}_out"
                    wire_declarations.append(f"    wire [{output_size-1}:0] {wire_name};")
                    layer_outputs.append((wire_name, None, output_size, False))  # No register
            else:
                # Final layer
                if layer_num in pipeline_positions:
                    # Output is registered
                    wire_name = "out_comb"
                    wire_declarations.append(f"    wire [{output_size-1}:0] {wire_name};")
                    register_code.append(generate_pipeline_register("out", output_size, wire_name))
                    layer_outputs.append((wire_name, "out", output_size, True))
                else:
                    # Output is combinational
                    layer_outputs.append(("out", None, output_size, False))

        # Add wire declarations
        body_lines.extend(wire_declarations)
        body_lines.append("")

        # Generate code for each layer
        for idx, (layer_type, layer_idx, output_shape, output_size) in enumerate(layer_info):
            comb_output = layer_outputs[idx][0]  # Combinational output name

            if layer_type == 'linear':
                # LogicDense layer
                layer_a, layer_b, layer_op = self.linear_layers[layer_idx]
                layer_code = self._get_linear_layer_verilog(
                    layer_a, layer_b, layer_op,
                    layer_id=idx,
                    input_name=current_input,
                    output_name=comb_output
                )
                body_lines.extend(layer_code)

            elif layer_type == 'conv':
                # LogicConv2d/3d layer
                conv_info = self.conv_layers[layer_idx]
                layer_code = self._get_conv_layer_verilog(
                    conv_info,
                    layer_id=idx,
                    input_name=current_input,
                    output_name=comb_output
                )
                body_lines.extend(layer_code)

            elif layer_type == 'pool':
                # OrPooling layer - implement as OR reduction
                body_lines.append(format_verilog_comment(f"Layer {idx}: OrPooling (max pooling)"))
                body_lines.append(format_verilog_comment("Note: Pooling not yet implemented in Verilog"))
                # TODO: Implement pooling layer verilog generation

            elif layer_type == 'flatten':
                # Flatten is a no-op in Verilog (just wire renaming)
                body_lines.append(format_verilog_comment(f"Layer {idx}: Flatten"))
                body_lines.append(f"    assign {comb_output} = {current_input};")

            elif layer_type == 'groupsum':
                # GroupSum - implement as adder tree
                body_lines.append(format_verilog_comment(f"Layer {idx}: GroupSum"))
                body_lines.append(format_verilog_comment("Note: GroupSum not yet implemented in Verilog"))
                # TODO: Implement GroupSum verilog generation

            # Update current_input for next layer
            # Use registered output if this layer has a register, otherwise use combinational
            if layer_outputs[idx][3]:  # Has register
                current_input = layer_outputs[idx][1]  # Use registered name
            else:
                current_input = layer_outputs[idx][0]  # Use combinational name

        # Add pipeline registers
        if register_code:
            body_lines.append("")
            body_lines.append(format_verilog_comment("Pipeline registers"))
            body_lines.extend(register_code)

        # Generate complete module
        output_is_reg = len(pipeline_positions) > 0 and num_layers in pipeline_positions
        module_code = generate_verilog_module(
            module_name=module_name,
            input_width=input_size,
            output_width=output_size,
            body='\n'.join(body_lines),
            add_clock=add_clock,
            output_registered=output_is_reg
        )

        return module_code

    def export_hdl(self, output_dir: str, module_name: str = "torchlogix_net",
                   format: str = "verilog", pipeline_stages: int = 0) -> None:
        """Export the network as HDL files.

        Args:
            output_dir: Directory to write HDL files
            module_name: Name of the top-level module
            format: HDL format - "verilog" or "vhdl" (only verilog supported currently)
            pipeline_stages: Number of pipeline stages (0 = combinational, see get_verilog_code)
        """
        import os

        if format.lower() not in ["verilog", "v"]:
            raise ValueError(f"Only 'verilog' format is currently supported, got '{format}'")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate Verilog code
        verilog_code = self.get_verilog_code(module_name=module_name,
                                             pipeline_stages=pipeline_stages)

        # Write to file
        output_path = os.path.join(output_dir, f"{module_name}.v")
        with open(output_path, 'w') as f:
            f.write(verilog_code)

        print(f"Verilog module written to: {output_path}")
        if pipeline_stages > 0:
            print(f"  Pipeline stages: {pipeline_stages}")
            print(f"  Latency: {pipeline_stages} clock cycles")

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
        if self.apply_thresholding:
            is_float = self.thresholding_layer["is_float"]
            inp_type = ctypes.c_float if is_float else BITS_TO_C_DTYPE[32]
        else:
            inp_type = ctypes.c_bool

        # Output type: float if scaling is applied, int otherwise
        out_type = ctypes.c_float if self.apply_groupsum_scaling else BITS_TO_C_DTYPE[32]

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

        # Output dtype: float32 if scaling is applied, int32 otherwise
        out_dtype = np.float32 if self.apply_groupsum_scaling else BITS_TO_NP_DTYPE[32]
        out = np.zeros(x.shape[0] * self.num_classes, dtype=out_dtype)
        x = x.reshape(-1)

        # Ensure input has the correct dtype for the library function
        if self.apply_thresholding:
            is_float = self.thresholding_layer["is_float"]
            x = x.astype(np.float32 if is_float else np.int32)
        else:
            x = x.astype(np.bool_)

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
