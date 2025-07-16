import ctypes
import math
import shutil
import subprocess
import tempfile
import time
from typing import Union

import numpy as np
import numpy.typing
import torch

from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer
from neurodifflogic.models.difflog_layers.conv import LogicConv2d, OrPoolingLayer

ALL_OPERATIONS = [
    "zero",
    "and",
    "not_implies",
    "a",
    "not_implied_by",
    "b",
    "xor",
    "or",
    "not_or",
    "not_xor",
    "not_b",
    "implied_by",
    "not_a",
    "implies",
    "not_and",
    "one",
]

BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0", 16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1", 16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {
    8: ctypes.c_int8,
    16: ctypes.c_int16,
    32: ctypes.c_int32,
    64: ctypes.c_int64,
}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


class CompiledLogicNet(torch.nn.Module):
    """
    A compiled logic net.
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        device="cpu",
        num_bits=64,
        cpu_compiler="gcc",
        verbose=False,
    ):
        super(CompiledLogicNet, self).__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.cpu_compiler = cpu_compiler

        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [8, 16, 32, 64]

        if self.model is not None:
            layers = []

            self.num_inputs = None

            assert isinstance(self.model[-1], GroupSum), (
                "The last layer of the model must be GroupSum, but it is {} / {}"
                " instead.".format(type(self.model[-1]), self.model[-1])
            )
            self.num_classes = self.model[-1].k

            first = True
            for layer in self.model:
                if isinstance(layer, LogicLayer):
                    if first:
                        self.num_inputs = layer.in_dim
                        first = False
                    self.num_out_per_class = layer.out_dim // self.num_classes
                    layers.append(
                        (layer.indices[0], layer.indices[1], layer.weight.argmax(1))
                    )
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose:
                        print(
                            "Skipping torch.nn.Flatten layer ({}).".format(type(layer))
                        )
                elif isinstance(layer, GroupSum):
                    if verbose:
                        print("Skipping GroupSum layer ({}).".format(type(layer)))
                else:
                    assert False, "Error: layer {} / {} unknown.".format(
                        type(layer), layer
                    )

            self.layers = layers

            if verbose:
                print("`layers` created and has {} layers.".format(len(layers)))

        self.lib_fn = None

    def get_gate_code(self, var1, var2, gate_op):
        """
        Get the code for a gate.
        """
        operation_name = ALL_OPERATIONS[gate_op]

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
            assert False, "Operator {} unknown.".format(operation_name)

        if self.num_bits == 8:
            res = f"(char) ({res})"
        elif self.num_bits == 16:
            res = f"(short) ({res})"

        return res

    def get_layer_code(self, layer_a, layer_b, layer_op, layer_id, prefix_sums):
        code = []
        for var_id, (gate_a, gate_b, gate_op) in enumerate(
            zip(layer_a, layer_b, layer_op)
        ):
            if layer_id == 0:
                a = f"inp[{gate_a}]"
                b = f"inp[{gate_b}]"
            else:
                a = f"v{prefix_sums[layer_id - 1] + gate_a}"
                b = f"v{prefix_sums[layer_id - 1] + gate_b}"
            if self.device == "cpu" and layer_id == len(prefix_sums) - 1:
                code.append(f"\tout[{var_id}] = {self.get_gate_code(a, b, gate_op)};")
            else:
                assert not (
                    self.device == "cpu" and layer_id >= len(prefix_sums) - 1
                ), (layer_id, len(prefix_sums))
                code.append(
                    f"\tconst {BITS_TO_DTYPE[self.num_bits]} "
                    f"v{prefix_sums[layer_id] + var_id} = "
                    f"{self.get_gate_code(a, b, gate_op)};"
                )
        return code

    def get_c_code(self):
        prefix_sums = [0]
        cur_count = 0
        for layer_a, layer_b, layer_op in self.layers[:-1]:
            cur_count += len(layer_a)
            prefix_sums.append(cur_count)

        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "",
            f"void logic_gate_net("
            f"{BITS_TO_DTYPE[self.num_bits]} const *inp, "
            f"{BITS_TO_DTYPE[self.num_bits]} *out);",
            "",
            f"void logic_gate_net("
            f"{BITS_TO_DTYPE[self.num_bits]} const *inp, "
            f"{BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ]

        for layer_id, (layer_a, layer_b, layer_op) in enumerate(self.layers):
            code.extend(
                self.get_layer_code(layer_a, layer_b, layer_op, layer_id, prefix_sums)
            )

        code.append("}")

        num_neurons_ll = self.layers[-1][0].shape[0]
        log2_of_num_neurons_per_class_ll = math.ceil(
            math.log2(num_neurons_ll / self.num_classes + 1)
        )

        code.append(
            rf"""
void apply_logic_gate_net(bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc({self.num_inputs}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc({num_neurons_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp_o = malloc({log2_of_num_neurons_per_class_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));

    for(size_t i = 0; i < len; ++i) {{

        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < {self.num_inputs}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(inp[i * {self.num_inputs} * {self.num_bits} + ({self.num_bits} - b - 1) * {self.num_inputs} + d]);
            }}
            inp_temp[d] = res;
        }}

        // Applying the logic gate net
        logic_gate_net(inp_temp, out_temp);

        // GroupSum of the results via logic gate networks
        for(size_t c = 0; c < {self.num_classes}; ++c) {{  // for each class
            // Initialize the output bits
            for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                out_temp_o[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            }}

            // Apply the adder logic gate network
            for(size_t a = 0; a < {self.layers[-1][0].shape[0] // self.num_classes}; ++a) {{
                {BITS_TO_DTYPE[self.num_bits]} carry = out_temp[c * {self.layers[-1][0].shape[0] // self.num_classes} + a];
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
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = res;
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}
"""
        )

        return "\n".join(code)

    def compile(self, opt_level=1, save_lib_path=None, verbose=False):
        """
        Regarding the optimization level for C compiler:

        compilation time vs. call time for 48k lines of code
        -O0 -> 5.5s compiling -> 269ms call
        -O1 -> 190s compiling -> 125ms call
        -O2 -> 256s compiling -> 130ms call
        -O3 -> 346s compiling -> 124ms call

        :param opt_level: optimization level for C compiler
        :param save_lib_path: (optional) where to save the .so shared object library
        :param verbose:
        :return:
        """

        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".c" if self.device != "cuda" else ".cu"
            ) as c_file:
                if self.device == "cpu":
                    code = self.get_c_code()
                else:
                    assert False, "Device {} not supported.".format(self.device)

                if verbose and len(code.split("\n")) <= 200:
                    print()
                    print()
                    print(code)
                    print()
                    print()

                c_file.write(code)
                c_file.flush()

                if verbose:
                    print(
                        "C code created and has {} lines. (temp location {})".format(
                            len(code.split("\n")), c_file.name
                        )
                    )

                t_s = time.time()
                if self.device == "cpu":
                    compiler_out = subprocess.run(
                        [
                            self.cpu_compiler,
                            "-shared",
                            "-fPIC",
                            "-O{}".format(opt_level),
                            # "-march=native",  # removed for compatibility with Apple Silicon: https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
                            "-o",
                            lib_file.name,
                            c_file.name,
                        ]
                    )
                else:
                    assert False, "Device {} not supported.".format(self.device)

                if compiler_out.returncode != 0:
                    raise RuntimeError(
                        f"compilation exited with error code {compiler_out.returncode}"
                    )

                print("Compiling finished in {:.3f} seconds.".format(time.time() - t_s))

            if save_lib_path is not None:
                shutil.copy(lib_file.name, save_lib_path)
                if verbose:
                    print(
                        "lib_file copied from {} to {} .".format(
                            lib_file.name, save_lib_path
                        )
                    )

            lib = ctypes.cdll.LoadLibrary(lib_file.name)
            # lib = ctypes.cdll.LoadLibrary('my-adder.so')

            lib_fn = lib.apply_logic_gate_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]

        self.lib_fn = lib_fn

    @staticmethod
    def load(save_lib_path, num_classes, num_bits):
        self = CompiledLogicNet(None, num_bits=num_bits)
        self.num_classes = num_classes

        lib = ctypes.cdll.LoadLibrary(save_lib_path)

        lib_fn = lib.apply_logic_gate_net
        lib_fn.restype = None
        lib_fn.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]

        self.lib_fn = lib_fn
        return self

    def forward(
        self,
        x: Union[torch.BoolTensor, numpy.typing.NDArray[np.bool_]],
        verbose: bool = False,
    ) -> torch.IntTensor:
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
        pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
        x = np.concatenate([x, np.zeros_like(x[:pad_len])])

        if verbose:
            print("x.shape", x.shape)

        out = np.zeros(x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32])
        x = x.reshape(-1)

        self.lib_fn(x, out, batch_size_div_bits)

        out = torch.tensor(out).view(
            batch_size_div_bits * self.num_bits, self.num_classes
        )
        if pad_len > 0:
            out = out[:-pad_len]
        if verbose:
            print("out.shape", out.shape)

        return out

import ctypes
import math
import shutil
import subprocess
import tempfile
import time
from typing import Union, Tuple

import numpy as np
import numpy.typing
import torch

from neurodifflogic.models.difflog_layers.conv import LogicConv2d
from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer

ALL_OPERATIONS = [
    "zero", "and", "not_implies", "a", "not_implied_by", "b", "xor", "or",
    "not_or", "not_xor", "not_b", "implied_by", "not_a", "implies", "not_and", "one",
]

BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0", 16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1", 16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {
    8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64,
}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


class CompiledConvLogicNet(torch.nn.Module):
    """
    A compiled convolutional logic network.
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        device="cpu",
        num_bits=64,
        cpu_compiler="gcc",
        verbose=False,
    ):
        super(CompiledConvLogicNet, self).__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.cpu_compiler = cpu_compiler

        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [8, 16, 32, 64]

        if self.model is not None:
            self.conv_layers = []
            self.linear_layers = []
            self.num_classes = None
            self.input_shape = None
            
            # Parse the model structure
            for layer in self.model:
                if isinstance(layer, LogicConv2d):
                    self.conv_layers.append(self._extract_conv_layer_info(layer))
                    if self.input_shape is None:
                        self.input_shape = (layer.channels, layer.in_dim[0], layer.in_dim[1])
                elif isinstance(layer, LogicLayer):
                    self.linear_layers.append(
                        (layer.indices[0], layer.indices[1], layer.weight.argmax(1))
                    )
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose:
                        print(f"Skipping torch.nn.Flatten layer ({type(layer)})")
                elif isinstance(layer, GroupSum):
                    self.num_classes = layer.k
                    if verbose:
                        print(f"Found GroupSum layer with {layer.k} classes")
                else:
                    if verbose:
                        print(f"Unknown layer type: {type(layer)}")

            if verbose:
                print(f"Parsed {len(self.conv_layers)} conv layers and {len(self.linear_layers)} linear layers")

        self.lib_fn = None

    def _extract_conv_layer_info(self, layer: LogicConv2d):
        """Extract the necessary information from a LogicConv2d layer for compilation."""
        # Get the learned operations for each level of the binary tree
        tree_operations = []
        for level_idx, level_weights in enumerate(layer.tree_weights):
            level_ops = []
            for weight_param in level_weights:
                ops = weight_param.argmax(1).cpu().numpy()  # Shape: (num_kernels,)
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

    def get_gate_code(self, var1, var2, gate_op):
        """Get the code for a gate operation."""
        operation_name = ALL_OPERATIONS[gate_op]

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
            assert False, f"Operator {operation_name} unknown."

        if self.num_bits == 8:
            res = f"(char) ({res})"
        elif self.num_bits == 16:
            res = f"(short) ({res})"

        return res

    def get_conv_layer_code(self, conv_info, layer_id):
        """Generate C code for a convolutional layer."""
        code = []
        indices = conv_info['indices']
        tree_ops = conv_info['tree_operations']
        
        # Calculate output dimensions
        h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                 // conv_info['stride']) + 1
        w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                 // conv_info['stride']) + 1
        
        code.append(f"\t// Convolutional layer {layer_id}")
        
        # Process each kernel
        for kernel_idx in range(conv_info['num_kernels']):
            code.append(f"\t// Kernel {kernel_idx}")
            
            # Process each output position
            for pos_idx in range(h_out * w_out):
                h_pos = pos_idx // w_out
                w_pos = pos_idx % w_out
                
                # First level: process receptive field positions
                level_0_indices = indices[0]
                left_indices = level_0_indices[0][kernel_idx, pos_idx]  # Shape: (2**tree_depth, 3)
                right_indices = level_0_indices[1][kernel_idx, pos_idx]  # Shape: (2**tree_depth, 3)
                
                # Generate variables for the first level
                for gate_idx in range(2**conv_info['tree_depth']):
                    left_h, left_w, left_c = left_indices[gate_idx]
                    right_h, right_w, right_c = right_indices[gate_idx]
                    
                    # Get operation for this gate - tree_ops[level][gate_idx] gives operations for all kernels
                    gate_op = tree_ops[0][gate_idx][kernel_idx]
                    
                    if layer_id == 0:
                        # Input layer - use channel-height-width indexing
                        left_var = f"inp[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {left_h} * {conv_info['in_dim'][1]} + {left_w}]"
                        right_var = f"inp[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {right_h} * {conv_info['in_dim'][1]} + {right_w}]"
                    else:
                        # Previous layer output
                        left_var = f"layer_{layer_id-1}_out[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {left_h} * {conv_info['in_dim'][1]} + {left_w}]"
                        right_var = f"layer_{layer_id-1}_out[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {right_h} * {conv_info['in_dim'][1]} + {right_w}]"
                    
                    var_name = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l0_g{gate_idx}"
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
                        
                        # Get operation for this gate
                        gate_op = tree_ops[level][gate_idx][kernel_idx]
                        
                        left_var = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l{level-1}_g{left_idx}"
                        right_var = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l{level-1}_g{right_idx}"
                        
                        if level == conv_info['tree_depth']:
                            # Final level - store in output
                            output_idx = (kernel_idx * h_out * w_out + pos_idx)
                            code.append(
                                f"\tlayer_{layer_id}_out[{output_idx}] = "
                                f"{self.get_gate_code(left_var, right_var, gate_op)};"
                            )
                        else:
                            var_name = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l{level}_g{gate_idx}"
                            code.append(
                                f"\tconst {BITS_TO_DTYPE[self.num_bits]} {var_name} = "
                                f"{self.get_gate_code(left_var, right_var, gate_op)};"
                            )
        
        return code

    def get_linear_layer_code(self, layer_a, layer_b, layer_op, layer_id, is_final=False):
        """Generate C code for a linear layer (same as original implementation)."""
        code = []
        for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
            a = f"linear_input[{gate_a}]"
            b = f"linear_input[{gate_b}]"
            
            if is_final:
                code.append(f"\tout[{var_id}] = {self.get_gate_code(a, b, gate_op)};")
            else:
                code.append(
                    f"\tconst {BITS_TO_DTYPE[self.num_bits]} "
                    f"linear_{layer_id}_out_{var_id} = "
                    f"{self.get_gate_code(a, b, gate_op)};"
                )
        return code

    def get_c_code(self):
        """Generate the complete C code for the network."""
        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "#include <string.h>",  # for memcpy
            "",
        ]

        # Calculate dimensions
        input_size = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        
        # Calculate sizes for each conv layer
        conv_output_sizes = []
        current_h, current_w = self.input_shape[1], self.input_shape[2]
        
        for conv_info in self.conv_layers:
            h_out = ((current_h + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                     // conv_info['stride']) + 1
            w_out = ((current_w + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                     // conv_info['stride']) + 1
            output_size = conv_info['num_kernels'] * h_out * w_out
            conv_output_sizes.append(output_size)
            current_h, current_w = h_out, w_out

        # Function signature
        final_output_size = self.linear_layers[-1][0].shape[0] if self.linear_layers else conv_output_sizes[-1]
        code.extend([
            f"void logic_conv_net("
            f"{BITS_TO_DTYPE[self.num_bits]} const *inp, "
            f"{BITS_TO_DTYPE[self.num_bits]} *out);",
            "",
            f"void logic_conv_net("
            f"{BITS_TO_DTYPE[self.num_bits]} const *inp, "
            f"{BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ])

        # Allocate intermediate buffers
        for i, size in enumerate(conv_output_sizes):
            code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} layer_{i}_out[{size}];")
        
        if self.linear_layers:
            code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_input[{conv_output_sizes[-1]}];")

        code.append("")

        # Generate conv layers
        for i, conv_info in enumerate(self.conv_layers):
            code.extend(self.get_conv_layer_code(conv_info, i))
            code.append("")

        # Copy conv output to linear input if needed
        if self.linear_layers and conv_output_sizes:
            code.append(f"\t// Copy conv output to linear input")
            code.append(f"\tmemcpy(linear_input, layer_{len(self.conv_layers)-1}_out, "
                       f"{conv_output_sizes[-1]} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
            code.append("")

        # Generate linear layers
        for i, (layer_a, layer_b, layer_op) in enumerate(self.linear_layers):
            is_final = (i == len(self.linear_layers) - 1)
            code.extend(self.get_linear_layer_code(layer_a, layer_b, layer_op, i, is_final))
            
            if not is_final:
                # Copy output to input for next layer
                code.append(f"\t// Copy to next layer input")
                for j in range(len(layer_a)):
                    code.append(f"\tlinear_input[{j}] = linear_{i}_out_{j};")

        # If no linear layers, copy conv output to final output
        if not self.linear_layers and conv_output_sizes:
            code.append(f"\t// Copy conv output to final output")
            code.append(f"\tmemcpy(out, layer_{len(self.conv_layers)-1}_out, "
                       f"{conv_output_sizes[-1]} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")

        code.append("}")

        # Add the batch processing function (similar to original)
        if self.num_classes:
            self._add_batch_processing_function(code, input_size, final_output_size)

        return "\n".join(code)

    def _add_batch_processing_function(self, code, input_size, output_size):
        """Add the batch processing function."""
        num_neurons_ll = output_size
        log2_of_num_neurons_per_class_ll = math.ceil(
            math.log2(num_neurons_ll / self.num_classes + 1)
        )

        code.append(
            rf"""
void apply_logic_conv_net(bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{
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

        // Applying the logic conv net
        logic_conv_net(inp_temp, out_temp);

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
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = res;
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}
"""
        )

    def compile(self, opt_level=1, save_lib_path=None, verbose=False):
        """Compile the network to a shared library."""
        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c") as c_file:
                code = self.get_c_code()

                if verbose and len(code.split("\n")) <= 200:
                    print("\n" + code + "\n")

                c_file.write(code)
                c_file.flush()

                if verbose:
                    print(f"C code created with {len(code.split('\n'))} lines. (temp location {c_file.name})")

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
            
            if self.num_classes:
                lib_fn = lib.apply_logic_conv_net
                lib_fn.restype = None
                lib_fn.argtypes = [
                    np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                    ctypes.c_size_t,
                ]
            else:
                lib_fn = lib.logic_conv_net
                lib_fn.restype = None
                lib_fn.argtypes = [
                    np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[self.num_bits], flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[self.num_bits], flags="C_CONTIGUOUS"),
                ]

        self.lib_fn = lib_fn

    @staticmethod
    def load(save_lib_path, input_shape, num_classes=None, num_bits=64):
        """Load a compiled network from a shared library."""
        self = CompiledConvLogicNet(None, num_bits=num_bits)
        self.input_shape = input_shape
        self.num_classes = num_classes

        lib = ctypes.cdll.LoadLibrary(save_lib_path)

        if num_classes:
            lib_fn = lib.apply_logic_conv_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]
        else:
            lib_fn = lib.logic_conv_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[self.num_bits], flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[self.num_bits], flags="C_CONTIGUOUS"),
            ]

        self.lib_fn = lib_fn
        return self

    def forward(
        self,
        x: Union[torch.BoolTensor, numpy.typing.NDArray[np.bool_]],
        verbose: bool = False,
    ) -> torch.IntTensor:
        """Forward pass through the compiled network."""
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if self.num_classes:
            # Use batch processing function
            batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
            pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
            x = np.concatenate([x, np.zeros_like(x[:pad_len])])

            if verbose:
                print("x.shape", x.shape)

            out = np.zeros(x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32])
            x = x.reshape(-1)

            self.lib_fn(x, out, batch_size_div_bits)

            out = torch.tensor(out).view(batch_size_div_bits * self.num_bits, self.num_classes)
            if pad_len > 0:
                out = out[:-pad_len]
        else:
            # Direct function call
            batch_size = x.shape[0]
            input_size = np.prod(self.input_shape)
            x_flat = x.reshape(batch_size, input_size).astype(BITS_TO_NP_DTYPE[self.num_bits])
            
            # Calculate output size based on network structure
            output_size = self._calculate_output_size()
            out = np.zeros((batch_size, output_size), dtype=BITS_TO_NP_DTYPE[self.num_bits])
            
            for i in range(batch_size):
                self.lib_fn(x_flat[i], out[i])
            
            out = torch.tensor(out)

        if verbose:
            print("out.shape", out.shape)

        return out

    def _calculate_output_size(self):
        """Calculate the output size of the network."""
        if self.linear_layers:
            return self.linear_layers[-1][0].shape[0]
        elif self.conv_layers:
            # Calculate the output size of the last conv layer
            conv_info = self.conv_layers[-1]
            h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                     // conv_info['stride']) + 1
            w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                     // conv_info['stride']) + 1
            return conv_info['num_kernels'] * h_out * w_out
        else:
            return 1  # Fallback
        



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

from neurodifflogic.models.difflog_layers.conv import LogicConv2d
from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer

ALL_OPERATIONS = [
    "zero", "and", "not_implies", "a", "not_implied_by", "b", "xor", "or",
    "not_or", "not_xor", "not_b", "implied_by", "not_a", "implies", "not_and", "one",
]

BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0", 16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1", 16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {
    8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64,
}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


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

from neurodifflogic.models.difflog_layers.conv import LogicConv2d
from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer

ALL_OPERATIONS = [
    "zero", "and", "not_implies", "a", "not_implied_by", "b", "xor", "or",
    "not_or", "not_xor", "not_b", "implied_by", "not_a", "implies", "not_and", "one",
]

BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0", 16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1", 16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {
    8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64,
}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


class CompiledCombinedLogicNet(torch.nn.Module):
    """
    Unified compiled logic network that handles both convolutional and linear layers.
    """

    def __init__(
        self,
        model: torch.nn.Sequential = None,
        device: str = "cpu",
        num_bits: int = 64,
        cpu_compiler: str = "gcc",
        verbose: bool = False,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.cpu_compiler = cpu_compiler

        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [8, 16, 32, 64]

        # Initialize layer storage
        self.conv_layers = []
        self.linear_layers = []
        self.num_classes = None
        self.input_shape = None
        self.lib_fn = None

        if model is not None:
            self._parse_model(verbose)

    def _parse_model(self, verbose: bool):
        """Parse the model structure, handling both conv and linear layers."""
        # Find GroupSum layer for num_classes
        for layer in self.model:
            if isinstance(layer, GroupSum):
                self.num_classes = layer.k
                break

        # Parse all layers
        for layer in self.model:
            if isinstance(layer, LogicConv2d):
                self.conv_layers.append(self._extract_conv_layer_info(layer))
                if self.input_shape is None:
                    self.input_shape = (layer.channels, layer.in_dim[0], layer.in_dim[1])
            elif isinstance(layer, LogicLayer):
                self.linear_layers.append(
                    (layer.indices[0], layer.indices[1], layer.weight.argmax(1))
                )
            elif isinstance(layer, torch.nn.Flatten):
                if verbose:
                    print(f"Skipping torch.nn.Flatten layer ({type(layer)})")
            elif isinstance(layer, GroupSum):
                if verbose:
                    print(f"Found GroupSum layer with {layer.k} classes")
            else:
                if verbose:
                    print(f"Warning: Unknown layer type: {type(layer)}")

        if verbose:
            print(f"Parsed {len(self.conv_layers)} conv layers and {len(self.linear_layers)} linear layers")

        # Validate model structure
        if not self.conv_layers and not self.linear_layers:
            raise ValueError("Model must contain at least one LogicConv2d or LogicLayer")
        
        # Set input shape for linear-only models
        if not self.conv_layers and self.linear_layers:
            # For linear-only models, assume flattened input
            first_linear = None
            for layer in self.model:
                if isinstance(layer, LogicLayer):
                    first_linear = layer
                    break
            if first_linear:
                self.input_shape = (first_linear.in_dim,)  # Flattened input

    def _extract_conv_layer_info(self, layer: LogicConv2d) -> Dict[str, Any]:
        """Extract information from a LogicConv2d layer for compilation."""
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

    def get_gate_code(self, var1: str, var2: str, gate_op: int) -> str:
        """Generate C code for a logic gate operation."""
        operation_name = ALL_OPERATIONS[gate_op]

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

        return res

    def _get_conv_layer_code(self, conv_info: Dict[str, Any], layer_id: int) -> List[str]:
        """Generate C code for a convolutional layer."""
        code = []
        indices = conv_info['indices']
        tree_ops = conv_info['tree_operations']

        h_out = ((conv_info['in_dim'][0] + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                 // conv_info['stride']) + 1
        w_out = ((conv_info['in_dim'][1] + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                 // conv_info['stride']) + 1

        code.append(f"\t// Convolutional layer {layer_id}")

        for kernel_idx in range(conv_info['num_kernels']):
            for pos_idx in range(h_out * w_out):
                # First level: process receptive field positions
                level_0_indices = indices[0]
                left_indices = level_0_indices[0][kernel_idx, pos_idx]
                right_indices = level_0_indices[1][kernel_idx, pos_idx]

                # Generate variables for the first level
                for gate_idx in range(2**conv_info['tree_depth']):
                    left_h, left_w, left_c = left_indices[gate_idx]
                    right_h, right_w, right_c = right_indices[gate_idx]

                    gate_op = tree_ops[0][gate_idx][kernel_idx]

                    if layer_id == 0:
                        left_var = f"inp[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {left_h} * {conv_info['in_dim'][1]} + {left_w}]"
                        right_var = f"inp[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {right_h} * {conv_info['in_dim'][1]} + {right_w}]"
                    else:
                        left_var = f"conv_layer_{layer_id-1}_out[{left_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {left_h} * {conv_info['in_dim'][1]} + {left_w}]"
                        right_var = f"conv_layer_{layer_id-1}_out[{right_c} * {conv_info['in_dim'][0]} * {conv_info['in_dim'][1]} + {right_h} * {conv_info['in_dim'][1]} + {right_w}]"

                    var_name = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l0_g{gate_idx}"
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

                        left_var = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l{level-1}_g{left_idx}"
                        right_var = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l{level-1}_g{right_idx}"

                        if level == conv_info['tree_depth']:
                            output_idx = (kernel_idx * h_out * w_out + pos_idx)
                            code.append(
                                f"\tconv_layer_{layer_id}_out[{output_idx}] = "
                                f"{self.get_gate_code(left_var, right_var, gate_op)};"
                            )
                        else:
                            var_name = f"conv_{layer_id}_k{kernel_idx}_p{pos_idx}_l{level}_g{gate_idx}"
                            code.append(
                                f"\tconst {BITS_TO_DTYPE[self.num_bits]} {var_name} = "
                                f"{self.get_gate_code(left_var, right_var, gate_op)};"
                            )

        return code

    def _get_linear_layer_code(self, layer_a, layer_b, layer_op, layer_id: int, is_final: bool, has_conv_layers: bool) -> List[str]:
        """Generate C code for a linear layer."""
        code = []
        
        # Determine input source based on layer position and model structure
        if has_conv_layers:
            # Conv -> Linear model: always use linear_input
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
        elif not has_conv_layers and len(self.linear_layers) > 1:
            # Multi-layer linear model: alternate between buffers
            if layer_id % 2 == 0:
                output_var = "linear_buf_a"
            else:
                output_var = "linear_buf_b"
        else:
            # This shouldn't happen, but fallback
            output_var = "out"
        
        for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
            a = f"{input_var}[{gate_a}]"
            b = f"{input_var}[{gate_b}]"

            code.append(f"\t{output_var}[{var_id}] = {self.get_gate_code(a, b, gate_op)};")

        return code

    def _calculate_conv_output_sizes(self) -> List[int]:
        """Calculate output sizes for each convolutional layer."""
        sizes = []
        current_h, current_w = self.input_shape[1], self.input_shape[2]

        for conv_info in self.conv_layers:
            h_out = ((current_h + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                     // conv_info['stride']) + 1
            w_out = ((current_w + 2 * conv_info['padding'] - conv_info['receptive_field_size']) 
                     // conv_info['stride']) + 1
            output_size = conv_info['num_kernels'] * h_out * w_out
            sizes.append(output_size)
            current_h, current_w = h_out, w_out

        return sizes

    def get_c_code(self) -> str:
        """Generate the complete C code for the network."""
        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "#include <string.h>",
            "",
            f"void logic_net("
            f"{BITS_TO_DTYPE[self.num_bits]} const *inp, "
            f"{BITS_TO_DTYPE[self.num_bits]} *out);",
            "",
            f"void logic_net("
            f"{BITS_TO_DTYPE[self.num_bits]} const *inp, "
            f"{BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ]

        # Calculate sizes
        conv_output_sizes = self._calculate_conv_output_sizes() if self.conv_layers else []
        
        # Allocate intermediate buffers for conv layers
        for i, size in enumerate(conv_output_sizes):
            code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} conv_layer_{i}_out[{size}];")

        # Allocate buffers for linear layers
        if self.linear_layers:
            if self.conv_layers:
                # Conv -> Linear: need input buffer
                code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_input[{conv_output_sizes[-1]}];")
            
            # For linear-only models with multiple layers, we need input/output buffers
            if not self.conv_layers and len(self.linear_layers) > 1:
                # We'll use two buffers and alternate between them
                max_layer_size = max(len(layer[0]) for layer in self.linear_layers)
                code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_buf_a[{max_layer_size}];")
                code.append(f"\t{BITS_TO_DTYPE[self.num_bits]} linear_buf_b[{max_layer_size}];")

        code.append("")

        # Generate conv layers
        for i, conv_info in enumerate(self.conv_layers):
            code.extend(self._get_conv_layer_code(conv_info, i))
            code.append("")

        # Copy conv output to linear input if needed
        if self.conv_layers and self.linear_layers:
            code.append(f"\t// Copy conv output to linear input")
            code.append(f"\tmemcpy(linear_input, conv_layer_{len(self.conv_layers)-1}_out, "
                       f"{conv_output_sizes[-1]} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
            code.append("")

        # Generate linear layers
        for i, (layer_a, layer_b, layer_op) in enumerate(self.linear_layers):
            is_final = (i == len(self.linear_layers) - 1)
            code.extend(self._get_linear_layer_code(layer_a, layer_b, layer_op, i, is_final, bool(self.conv_layers)))

            # For multi-layer linear networks, copy outputs between layers
            if not is_final and not self.conv_layers:
                layer_size = len(layer_a)
                next_layer_size = len(self.linear_layers[i + 1][0])
                
                # Determine which buffer to copy from/to
                if i % 2 == 0:
                    # Copy from buf_a to buf_b
                    code.append(f"\t// Copy layer {i} output (buf_a) to next layer input (buf_b)")
                    code.append(f"\tmemcpy(linear_buf_b, linear_buf_a, {layer_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")
                else:
                    # Copy from buf_b to buf_a  
                    code.append(f"\t// Copy layer {i} output (buf_b) to next layer input (buf_a)")
                    code.append(f"\tmemcpy(linear_buf_a, linear_buf_b, {layer_size} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")

        # If no linear layers, copy conv output to final output
        if self.conv_layers and not self.linear_layers:
            code.append(f"\t// Copy conv output to final output")
            code.append(f"\tmemcpy(out, conv_layer_{len(self.conv_layers)-1}_out, "
                       f"{conv_output_sizes[-1]} * sizeof({BITS_TO_DTYPE[self.num_bits]}));")

        code.append("}")

        # Add batch processing function if needed
        if self.num_classes:
            code.extend(self._generate_batch_processing_function())

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
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = res;
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}
"""
        ]

    def _get_input_size(self) -> int:
        """Get the total input size."""
        if self.input_shape is None:
            raise ValueError("Input shape not set")
        return np.prod(self.input_shape)

    def _get_output_size(self) -> int:
        """Get the total output size."""
        if self.linear_layers:
            return len(self.linear_layers[-1][0])
        elif self.conv_layers:
            conv_output_sizes = self._calculate_conv_output_sizes()
            return conv_output_sizes[-1]
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
                    print(f"C code created with {len(code.split('\n'))} lines. (temp location {c_file.name})")

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
        if self.num_classes:
            lib_fn = lib.apply_logic_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]
        else:
            lib_fn = lib.logic_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[self.num_bits], flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[self.num_bits], flags="C_CONTIGUOUS"),
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

        if self.num_classes:
            return self._forward_with_groupsum(x, verbose)
        else:
            return self._forward_direct(x, verbose)

    def _forward_with_groupsum(self, x: np.ndarray, verbose: bool) -> torch.IntTensor:
        """Forward pass with GroupSum (batch processing)."""
        batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
        pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
        x = np.concatenate([x, np.zeros_like(x[:pad_len])])

        if verbose:
            print("x.shape", x.shape)

        out = np.zeros(x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32])
        x = x.reshape(-1)

        self.lib_fn(x, out, batch_size_div_bits)

        out = torch.tensor(out).view(batch_size_div_bits * self.num_bits, self.num_classes)
        if pad_len > 0:
            out = out[:-pad_len]

        if verbose:
            print("out.shape", out.shape)

        return out

    def _forward_direct(self, x: np.ndarray, verbose: bool) -> torch.IntTensor:
        """Direct forward pass without GroupSum."""
        batch_size = x.shape[0]
        input_size = self._get_input_size()
        x_flat = x.reshape(batch_size, input_size).astype(BITS_TO_NP_DTYPE[self.num_bits])

        output_size = self._get_output_size()
        out = np.zeros((batch_size, output_size), dtype=BITS_TO_NP_DTYPE[self.num_bits])

        for i in range(batch_size):
            self.lib_fn(x_flat[i], out[i])

        return torch.tensor(out)

    @staticmethod
    def load(save_lib_path: str, input_shape: tuple, num_classes: int = None, num_bits: int = 64):
        """Load a compiled network from a shared library."""
        self = CompiledLogicNet(None, num_bits=num_bits)
        self.input_shape = input_shape
        self.num_classes = num_classes

        lib = ctypes.cdll.LoadLibrary(save_lib_path)
        self._setup_library_function(lib)
        return self


