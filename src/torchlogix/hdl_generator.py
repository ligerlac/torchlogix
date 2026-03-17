"""HDL generation utilities for torchlogix models.

This module provides functions for generating Verilog/VHDL code from trained
torchlogix models. It maps the 16 Boolean gate operations to HDL syntax and
provides templates for module generation.
"""

# 16 Boolean operations (matches functional.py and compiled_model.py)
ALL_OPERATIONS = [
    "zero", "and", "not_implies", "a", "not_implied_by", "b", "xor", "or",
    "not_or", "not_xor", "not_b", "implied_by", "not_a", "implies", "not_and", "one",
]

# Gate ID to Verilog operator mapping
# Maps gate_id (0-15) to Verilog expression template with {} placeholders for a, b
GATE_TO_VERILOG = {
    0:  "1'b0",                    # zero - constant 0
    1:  "({0} & {1})",              # and
    2:  "({0} & ~{1})",             # not_implies - A and not B
    3:  "{0}",                      # a - pass through A
    4:  "(~{0} & {1})",             # not_implied_by - not A and B
    5:  "{1}",                      # b - pass through B
    6:  "({0} ^ {1})",              # xor
    7:  "({0} | {1})",              # or
    8:  "~({0} | {1})",             # not_or - NOR
    9:  "~({0} ^ {1})",             # not_xor - XNOR
    10: "~{1}",                     # not_b - NOT B
    11: "(~{1} | {0})",             # implied_by - not B or A
    12: "~{0}",                     # not_a - NOT A
    13: "(~{0} | {1})",             # implies - not A or B
    14: "~({0} & {1})",             # not_and - NAND
    15: "1'b1",                    # one - constant 1
}

# Gate ID to VHDL operator mapping (for future VHDL support)
GATE_TO_VHDL = {
    0:  "'0'",                     # zero
    1:  "({0} and {1})",            # and
    2:  "({0} and not {1})",        # not_implies
    3:  "{0}",                      # a
    4:  "(not {0} and {1})",        # not_implied_by
    5:  "{1}",                      # b
    6:  "({0} xor {1})",            # xor
    7:  "({0} or {1})",             # or
    8:  "(not ({0} or {1}))",       # not_or
    9:  "(not ({0} xor {1}))",      # not_xor
    10: "(not {1})",                # not_b
    11: "(not {1} or {0})",         # implied_by
    12: "(not {0})",                # not_a
    13: "(not {0} or {1})",         # implies
    14: "(not ({0} and {1}))",      # not_and
    15: "'1'",                      # one
}


def gate_id_to_verilog(gate_id: int, var_a: str, var_b: str) -> str:
    """Convert a gate ID and two variable names to a Verilog expression.

    Args:
        gate_id: Gate operation ID (0-15)
        var_a: Name of first input variable
        var_b: Name of second input variable

    Returns:
        Verilog expression string

    Example:
        >>> gate_id_to_verilog(1, "inp[0]", "inp[1]")
        '(inp[0] & inp[1])'
        >>> gate_id_to_verilog(7, "a", "b")
        '(a | b)'
    """
    if gate_id < 0 or gate_id > 15:
        raise ValueError(f"Invalid gate_id: {gate_id}. Must be 0-15.")
    
    # handle pytorch tensor gate_id
    if hasattr(gate_id, 'item'):
        gate_id = gate_id.item()

    template = GATE_TO_VERILOG[gate_id]
    return template.format(var_a, var_b)


def gate_id_to_vhdl(gate_id: int, var_a: str, var_b: str) -> str:
    """Convert a gate ID and two variable names to a VHDL expression.
    
    Args:
        gate_id: Gate operation ID (0-15)
        var_a: Name of first input variable
        var_b: Name of second input variable
    
    Returns:
        VHDL expression string
    """
    # Handle pytorch tensor gate_id
    if hasattr(gate_id, 'item'):
        gate_id = gate_id.item()
    
    if gate_id not in GATE_TO_VHDL:
        raise ValueError(f"Invalid gate_id: {gate_id}. Must be 0-15.")
    
    return GATE_TO_VHDL[gate_id].format(var_a, var_b)


def generate_pipeline_register(reg_name: str, width: int, input_name: str,
                              reset_value: int = 0) -> str:
    """Generate Verilog code for a pipeline register.

    Args:
        reg_name: Name of the register output
        width: Width of the register in bits
        input_name: Name of the input signal to register
        reset_value: Reset value for the register (default: 0)

    Returns:
        Verilog code for the register
    """
    code = []
    code.append(f"    reg [{width-1}:0] {reg_name};")
    code.append(f"    always @(posedge clk) begin")
    code.append(f"        if (rst)")
    code.append(f"            {reg_name} <= {width}'d{reset_value};")
    code.append(f"        else")
    code.append(f"            {reg_name} <= {input_name};")
    code.append(f"    end")
    return '\n'.join(code)


def generate_verilog_module(module_name: str,
                            input_width: int,
                            output_width: int,
                            body: str,
                            add_clock: bool = False,
                            output_registered: bool = False) -> str:
    """Generate a Verilog module with the given body.

    Args:
        module_name: Name of the module
        input_width: Width of input bus in bits
        output_width: Width of output bus in bits
        body: Module body (wire declarations and assignments)
        add_clock: Whether to add clock and reset inputs
        output_registered: Whether output is a reg (vs wire)

    Returns:
        Complete Verilog module code
    """
    # Port declarations
    ports = []
    if add_clock:
        ports.append("input wire clk")
        ports.append("input wire rst")
    ports.append(f"input wire [{input_width-1}:0] inp")

    # Output can be wire or reg depending on design
    output_type = "reg" if output_registered else "wire"
    ports.append(f"output {output_type} [{output_width-1}:0] out")

    port_list = ",\n    ".join(ports)

    module_code = f"""module {module_name} (
    {port_list}
);

{body}

endmodule
"""
    return module_code


def generate_vhdl_entity(entity_name: str,
                        input_width: int,
                        output_width: int,
                        architecture_body: str,
                        add_clock: bool = False) -> str:
    """Generate a VHDL entity with the given architecture body.

    Args:
        entity_name: Name of the entity
        input_width: Width of input bus in bits
        output_width: Width of output bus in bits
        architecture_body: Architecture body (signal declarations and logic)
        add_clock: Whether to add clock and reset inputs

    Returns:
        Complete VHDL entity and architecture code
    """
    # Port declarations
    ports = []
    if add_clock:
        ports.append("        clk : in std_logic")
        ports.append("        rst : in std_logic")
    ports.append(f"        inp : in std_logic_vector({input_width-1} downto 0)")
    ports.append(f"        out : out std_logic_vector({output_width-1} downto 0)")

    port_list = ";\n".join(ports)

    entity_code = f"""library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity {entity_name} is
    Port (
{port_list}
    );
end {entity_name};

architecture Behavioral of {entity_name} is
{architecture_body}
end Behavioral;
"""
    return entity_code


def sanitize_wire_name(name: str) -> str:
    """Sanitize a wire name to be valid Verilog/VHDL.

    Replaces invalid characters with underscores.

    Args:
        name: Raw wire name

    Returns:
        Sanitized wire name
    """
    # Replace invalid characters with underscore
    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    return ''.join(c if c in valid_chars else '_' for c in name)


def format_verilog_comment(text: str) -> str:
    """Format a text string as a Verilog comment.

    Args:
        text: Comment text

    Returns:
        Formatted comment line
    """
    return f"// {text}"


def format_vhdl_comment(text: str) -> str:
    """Format a text string as a VHDL comment.

    Args:
        text: Comment text

    Returns:
        Formatted comment line
    """
    return f"-- {text}"


# Truth table for validation (maps gate_id to [AB=00, AB=01, AB=10, AB=11])
GATE_TRUTH_TABLES = {
    0:  [0, 0, 0, 0],  # zero
    1:  [0, 0, 0, 1],  # and
    2:  [0, 0, 1, 0],  # not_implies
    3:  [0, 0, 1, 1],  # a
    4:  [0, 1, 0, 0],  # not_implied_by
    5:  [0, 1, 0, 1],  # b
    6:  [0, 1, 1, 0],  # xor
    7:  [0, 1, 1, 1],  # or
    8:  [1, 0, 0, 0],  # not_or
    9:  [1, 0, 0, 1],  # not_xor
    10: [1, 0, 1, 0],  # not_b
    11: [1, 0, 1, 1],  # implied_by
    12: [1, 1, 0, 0],  # not_a
    13: [1, 1, 0, 1],  # implies
    14: [1, 1, 1, 0],  # not_and
    15: [1, 1, 1, 1],  # one
}
