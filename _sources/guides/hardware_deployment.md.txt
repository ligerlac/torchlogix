# Hardware Deployment Guide

This guide explains how to deploy TorchLogix models to FPGAs and other hardware platforms using direct Verilog/RTL generation.

## Overview

TorchLogix can generate hardware descriptions (Verilog RTL) directly from trained models, enabling deployment to FPGAs and ASIC implementations. This provides an alternative to the traditional C→HLS→RTL pipeline and offers several advantages for logic gate networks.

### Why Hardware Deployment?

**Benefits**:
- **Ultra-low latency**: FPGA implementations can achieve sub-microsecond inference
- **High throughput**: Massive parallelism enables processing thousands of inputs per second
- **Energy efficiency**: Specialized hardware is more power-efficient than general-purpose CPUs/GPUs
- **Deterministic timing**: Predictable performance for real-time applications

**Use Cases**:
- Low-latency inference at network edge
- Real-time signal processing and control systems
- High-throughput batch processing
- Embedded systems with strict power budgets

### Design Approach: Direct Gate-Level Synthesis

TorchLogix uses a **direct gate-level synthesis** approach rather than LUT-based truth tables:

```
TorchLogix Model (PyTorch)
         ↓
   Gate Operations (AND, OR, XOR, etc.)
         ↓
   Verilog Expressions
         ↓
   FPGA Synthesis
```

**Why direct gates instead of LUTs?**
1. **Natural mapping**: TorchLogix already defines 16 gate operations
2. **Better optimization**: Modern synthesis tools optimize gate-level HDL effectively
3. **More readable**: `assign out = a & b;` is clearer than case statements
4. **Binary structure**: Each neuron has exactly 2 inputs → perfect for direct gates

### Comparison: Direct Verilog vs C→HLS

| Aspect | Direct Verilog | C→HLS→RTL |
|--------|---------------|-----------|
| **Generation** | Direct from model | Compile C, then HLS |
| **Intermediate** | None | C code + HLS directives |
| **Control** | Full RTL control | HLS tool dependent |
| **Readability** | Gate-level, explicit | High-level C abstractions |
| **Use Case** | FPGA-specific deployment | Cross-platform (CPU + FPGA) |

Both approaches are supported by TorchLogix. Use direct Verilog for FPGA-specific optimization and C→HLS for flexibility.

### Supported Layer Types

| Layer Type | Verilog Support | Notes |
|------------|-----------------|-------|
| **LogicDense** | ✅ Fully Supported | Direct gate synthesis |
| **LogicConv2d** | ✅ Fully Supported | Binary tree structure |
| **LogicConv3d** | ✅ Fully Supported | Binary tree with 3D indexing |
| **Flatten** | ✅ Supported | Wire passthrough |
| **OrPooling** | ⚠️ TODO | Recognized but not yet generated |
| **GroupSum** | ⚠️ TODO | Recognized but not yet generated |

Models using unsupported layers can still generate Verilog for supported portions, or use C code generation as an alternative.

---

## Basic Verilog Export

### Quick Start

Generate Verilog from any trained TorchLogix model:

```python
import torch
import torch.nn as nn
from torchlogix.layers import LogicDense, GroupSum
from torchlogix import CompiledLogicNet

# Create or load your model
model = nn.Sequential(
    LogicDense(8, 32, connections="fixed", device="cpu"),
    LogicDense(32, 32, connections="fixed", device="cpu"),
    GroupSum(1, tau=1.0)
)

# Compile the model
compiled = CompiledLogicNet(
    model,
    input_shape=(8,),
    use_bitpacking=False,
    num_bits=1
)

# Generate Verilog code
verilog_code = compiled.get_verilog_code(module_name="my_logic_net")

# Export to file
compiled.export_hdl(
    output_dir="./verilog_output",
    module_name="my_logic_net",
    format="verilog"
)
```

### API Reference

#### `get_verilog_code(module_name, pipeline_stages)`

Generates complete Verilog module as a string.

**Parameters**:
- `module_name` (str): Name of the top-level Verilog module (default: `"torchlogix_net"`)
- `pipeline_stages` (int): Number of pipeline stages (default: `0`)
  - `0`: Fully combinational (no registers, 1 cycle latency)
  - `1`: Single output register (helps synthesis)
  - `N`: Divide layers into N pipeline stages (N cycle latency)
  - `len(layers)`: Full layer-level pipelining (highest fmax)

**Returns**: Complete Verilog code as string

#### `export_hdl(output_dir, module_name, format, pipeline_stages)`

Exports Verilog to a file.

**Parameters**:
- `output_dir` (str): Directory to write Verilog file
- `module_name` (str): Module name (default: `"torchlogix_net"`)
- `format` (str): HDL format, currently only `"verilog"` supported (default: `"verilog"`)
- `pipeline_stages` (int): Pipeline configuration (default: `0`)

**Creates**: `{output_dir}/{module_name}.v`

### Understanding Generated Verilog

#### Combinational Design (pipeline_stages=0)

For a simple 2-layer network:

```verilog
module logic_net (
    input wire [7:0] inp,
    output wire [3:0] out
);
    // No clock or reset needed

    // Layer 0: LogicDense (4 neurons)
    wire [3:0] layer_0_out;
    assign layer_0_out[0] = (inp[0] & inp[2]);  // AND gate
    assign layer_0_out[1] = (inp[1] | inp[3]);  // OR gate
    assign layer_0_out[2] = (inp[4] ^ inp[5]);  // XOR gate
    assign layer_0_out[3] = ~(inp[6] & inp[7]); // NAND gate

    // Layer 1: LogicDense (2 neurons)
    assign out[0] = (layer_0_out[0] | layer_0_out[1]);
    assign out[1] = (layer_0_out[2] ^ layer_0_out[3]);

endmodule
```

**Characteristics**:
- Pure combinational logic (no state)
- No clock or reset signals
- 1 cycle latency (output available same cycle as input)
- Critical path spans entire network

#### Pipelined Design (pipeline_stages=2)

With pipeline registers:

```verilog
module logic_net (
    input wire clk,
    input wire rst,
    input wire [7:0] inp,
    output reg [3:0] out
);
    // Combinational wires
    wire [3:0] layer_0_comb;
    wire [3:0] out_comb;

    // Pipeline register
    reg [3:0] layer_0_out;

    // Layer 0: Combinational logic
    assign layer_0_comb[0] = (inp[0] & inp[2]);
    assign layer_0_comb[1] = (inp[1] | inp[3]);
    assign layer_0_comb[2] = (inp[4] ^ inp[5]);
    assign layer_0_comb[3] = ~(inp[6] & inp[7]);

    // Pipeline register after Layer 0
    always @(posedge clk) begin
        if (rst)
            layer_0_out <= 4'd0;
        else
            layer_0_out <= layer_0_comb;
    end

    // Layer 1: Combinational logic
    assign out_comb[0] = (layer_0_out[0] | layer_0_out[1]);
    assign out_comb[1] = (layer_0_out[2] ^ layer_0_out[3]);

    // Output register
    always @(posedge clk) begin
        if (rst)
            out <= 4'd0;
        else
            out <= out_comb;
    end

endmodule
```

**Characteristics**:
- Synchronous design with clock and reset
- Registers break up long combinational paths
- N cycle latency (where N = pipeline_stages)
- Higher maximum frequency (fmax)

#### Gate Operations Supported

All 16 two-input Boolean operations are supported:

| Gate ID | Operation | Verilog Expression |
|---------|-----------|-------------------|
| 0 | Zero (constant) | `1'b0` |
| 1 | AND | `(a & b)` |
| 2 | A AND NOT B | `(a & ~b)` |
| 3 | A (passthrough) | `a` |
| 4 | NOT A AND B | `(~a & b)` |
| 5 | B (passthrough) | `b` |
| 6 | XOR | `(a ^ b)` |
| 7 | OR | `(a \| b)` |
| 8 | NOR | `~(a \| b)` |
| 9 | XNOR | `~(a ^ b)` |
| 10 | NOT B | `~b` |
| 11 | B IMPLIES A | `(~b \| a)` |
| 12 | NOT A | `~a` |
| 13 | A IMPLIES B | `(~a \| b)` |
| 14 | NAND | `~(a & b)` |
| 15 | One (constant) | `1'b1` |

---

## Pipelining for Large Models

### The Problem: Large Combinational Designs

By default, TorchLogix generates **fully combinational** Verilog where all logic executes in a single clock cycle. This works well for small models but causes serious problems for larger ones:

#### Symptoms of Combinational Overload
- Synthesis fails or runs for hours without completing
- Verilog files >1M lines take forever to process
- Very low maximum frequency (fmax < 50 MHz)
- Timing closure failures (negative WNS)
- "Design too large" errors from Vivado

#### Why This Happens
- Deep combinational paths through many layers
- Synthesis tools struggle to optimize very large logic cones
- Critical path delay grows with model depth
- No natural break points for timing optimization

### The Solution: Pipeline Stages

**Pipelining** inserts registers between layers to break up long combinational paths:

```
Combinational (pipeline_stages=0):
  Input → [Layer0 → Layer1 → Layer2 → Layer3] → Output
  All in 1 cycle, huge critical path

Pipelined (pipeline_stages=4):
  Input → [Layer0] → REG → [Layer1] → REG → [Layer2] → REG → [Layer3] → REG → Output
  4 cycles latency, short critical paths
```

#### Benefits
- Synthesis succeeds even for very large models
- Much faster synthesis time (minutes vs hours)
- Higher maximum frequency (200+ MHz vs <50 MHz)
- Predictable timing closure
- Better resource utilization

#### Trade-offs
- Increased latency (N cycles instead of 1)
- More flip-flops (registers consume area)
- Need to handle clock and reset signals

### Pipeline Stage Options

#### `pipeline_stages=0` - Fully Combinational (Default)

```python
verilog = compiled.get_verilog_code(pipeline_stages=0)
```

- No registers, no clock required
- 1 cycle latency
- **Use for:** Small models (<10 layers), initial prototyping
- **Avoid for:** Large models (synthesis will fail)

#### `pipeline_stages=1` - Output Register Only

```python
verilog = compiled.get_verilog_code(pipeline_stages=1)
```

- Single register at output
- 1 cycle latency
- **Use for:** Medium models where synthesis struggles but you need low latency
- **Best for:** 10-30 layer models

#### `pipeline_stages=N` - N Pipeline Stages

```python
# 4 pipeline stages
verilog = compiled.get_verilog_code(pipeline_stages=4)
```

- Layers divided into N groups, register after each group
- N cycle latency
- **Use for:** Large models (50-200 layers)
- **Best for:** Balancing latency vs synthesis speed

#### Full Layer-Level Pipelining

```python
# Register between every layer
num_layers = len([m for m in model.modules() if isinstance(m, (LogicDense, LogicConv2d))])
verilog = compiled.get_verilog_code(pipeline_stages=num_layers)

# Or just use a large number
verilog = compiled.get_verilog_code(pipeline_stages=999)
```

- Register after every single layer
- Maximum possible fmax
- Highest latency (= number of layers)
- **Use for:** Very large models (>200 layers) or maximum throughput applications

### Choosing the Right Pipeline Configuration

#### Decision Tree

```
Is synthesis failing or very slow?
│
├─ NO → Use pipeline_stages=0 (fully combinational)
│        Lowest latency, simplest design
│
└─ YES → How many layers in your model?
         │
         ├─ <20 layers → pipeline_stages=1
         │               (Output register only)
         │
         ├─ 20-100 layers → pipeline_stages=4 to 8
         │                  (Balanced approach)
         │
         └─ >100 layers → pipeline_stages=N/4 to N
                          (N = number of layers)
```

#### Size Guidelines

| Model Characteristics | Recommended Config | Latency | Benefits |
|-----------------------|--------------------|---------|----------|
| <10 layers, <100K Verilog lines | `pipeline_stages=0` | 1 cycle | Simple, low latency |
| 10-30 layers, synthesis slow | `pipeline_stages=1` | 1 cycle | Helps synthesis |
| 30-100 layers | `pipeline_stages=4` | 4 cycles | Good balance |
| 100-200 layers | `pipeline_stages=8-16` | 8-16 cycles | Reliable synthesis |
| >200 layers | `pipeline_stages=N/4` | N/4 cycles | Fast synthesis |
| Maximum throughput needed | `pipeline_stages=999` | N cycles | Highest fmax |

#### Empirical Testing

Start conservative and increase pipelining if needed:

```python
# Step 1: Try combinational
verilog = compiled.get_verilog_code(pipeline_stages=0)
# Try to synthesize... if it fails or is very slow:

# Step 2: Add output register
verilog = compiled.get_verilog_code(pipeline_stages=1)
# Try to synthesize... if still slow:

# Step 3: Increase stages
for stages in [2, 4, 8, 16]:
    verilog = compiled.get_verilog_code(pipeline_stages=stages)
    # Synthesize and check timing/area trade-off
```

### Performance Optimization

#### Finding Optimal Pipeline Depth

Run synthesis with different configurations and compare:

```python
import subprocess

results = []
for stages in [0, 1, 2, 4, 8, 16]:
    verilog = compiled.get_verilog_code(
        module_name=f'design_p{stages}',
        pipeline_stages=stages
    )

    # Save Verilog
    with open(f'design_p{stages}.v', 'w') as f:
        f.write(verilog)

    # Synthesize (see Synthesis section for details)
    subprocess.run([
        'vivado', '-mode', 'batch',
        '-source', 'synthesize.tcl',
        '-tclargs', f'design_p{stages}.v', 'xc7z020clg400-1'
    ])

    # Parse and compare results
    # results.append((stages, luts, ffs, fmax, synthesis_time))

# Find optimal trade-off based on your requirements
```

#### Common Issues

**Issue: Pipelined design has lower fmax than expected**
- **Cause:** Not enough pipeline stages, or uneven distribution
- **Solution:** Increase `pipeline_stages` or try full layer-level pipelining

**Issue: Too much area consumed by registers**
- **Cause:** Too many pipeline stages for the model size
- **Solution:** Reduce `pipeline_stages` to find balance

**Issue: Synthesis still slow with pipelining**
- **Cause:** Individual layers may still be very large
- **Solution:**
  - Check if conv layers with large receptive fields need breaking up
  - Use more pipeline stages
  - Consider model architecture changes

---

## Testing Generated Verilog

Functional testing and verification ensures your generated Verilog matches the expected behavior from the trained model.

### Prerequisites

You'll need one of the following simulators:
- **Vivado Simulator (xsim)** - Included with Vivado
- **ModelSim/QuestaSim** - Commercial simulator from Mentor/Siemens
- **Icarus Verilog** - Open-source, free (`apt install iverilog` or `brew install icarus-verilog`)
- **Verilator** - Fast open-source simulator (`apt install verilator` or `brew install verilator`)

### Step 1: Generate Test Vectors

Export test vectors from your trained model using Python:

```python
import torch
import numpy as np
from torchlogix import CompiledLogicNet

# Load your trained model
model = ...  # Your trained TorchLogix model

# Generate test vectors
compiled = CompiledLogicNet(model, input_shape=(8,), use_bitpacking=False, num_bits=1)
compiled.compile()

# Generate random binary test cases
num_tests = 100
input_size = 8  # Match your model's input size
test_inputs = np.random.randint(0, 2, (num_tests, input_size), dtype=np.int8)

# Get expected outputs
test_outputs = []
for inp in test_inputs:
    out = compiled.forward(inp.reshape(1, -1))
    test_outputs.append(out[0])

# Save to files for testbench
np.savetxt('test_vectors_input.txt', test_inputs, fmt='%d')
np.savetxt('test_vectors_output.txt', np.array(test_outputs), fmt='%d')

print(f"Generated {num_tests} test vectors")
```

### Step 2: Create a Verilog Testbench

Create a testbench file `tb_logic_net.v` for combinational designs:

```verilog
`timescale 1ns/1ps

module tb_logic_net;
    // Parameters
    parameter INPUT_WIDTH = 8;
    parameter OUTPUT_WIDTH = 2;
    parameter NUM_TESTS = 100;

    // Signals
    reg [INPUT_WIDTH-1:0] inp;
    wire [OUTPUT_WIDTH-1:0] out;

    // Expected output
    reg [OUTPUT_WIDTH-1:0] expected_out;

    // Test vectors
    reg [INPUT_WIDTH-1:0] test_inputs [0:NUM_TESTS-1];
    reg [OUTPUT_WIDTH-1:0] test_outputs [0:NUM_TESTS-1];

    integer i;
    integer errors;

    // Instantiate the DUT (Device Under Test)
    logic_net dut (
        .inp(inp),
        .out(out)
    );

    // Load test vectors
    initial begin
        $readmemb("test_vectors_input.txt", test_inputs);
        $readmemb("test_vectors_output.txt", test_outputs);
        errors = 0;
    end

    // Test stimulus
    initial begin
        $display("Starting testbench...");
        $display("Time\t\tInput\t\tOutput\t\tExpected\tStatus");
        $display("----\t\t-----\t\t------\t\t--------\t------");

        // Run through all test vectors
        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            inp = test_inputs[i];
            expected_out = test_outputs[i];
            #10;  // Wait 10ns for combinational logic to settle

            // Check output
            if (out !== expected_out) begin
                $display("%0t\t%b\t%b\t%b\t\tFAIL", $time, inp, out, expected_out);
                errors = errors + 1;
            end else begin
                $display("%0t\t%b\t%b\t%b\t\tPASS", $time, inp, out, expected_out);
            end
        end

        // Summary
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total tests: %0d", NUM_TESTS);
        $display("Passed:      %0d", NUM_TESTS - errors);
        $display("Failed:      %0d", errors);

        if (errors == 0) begin
            $display("\nALL TESTS PASSED!");
        end else begin
            $display("\nSOME TESTS FAILED!");
        end

        $finish;
    end

    // Optional: Generate VCD waveform dump
    initial begin
        $dumpfile("tb_logic_net.vcd");
        $dumpvars(0, tb_logic_net);
    end

endmodule
```

#### Testbench for Pipelined Designs

For pipelined designs, you need to account for latency:

```verilog
module tb_pipelined_logic_net;
    parameter PIPELINE_LATENCY = 4;  // Match your pipeline_stages

    reg clk = 0;
    reg rst = 1;
    reg [7:0] inp;
    wire [1:0] out;

    // Generate clock (100 MHz)
    always #5 clk = ~clk;

    logic_net dut (
        .clk(clk),
        .rst(rst),
        .inp(inp),
        .out(out)
    );

    initial begin
        // Reset sequence
        rst = 1;
        #20 rst = 0;  // Release reset after 2 cycles

        // Run tests with pipeline latency
        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            inp = test_inputs[i];

            // Wait for pipeline to fill
            repeat(PIPELINE_LATENCY) @(posedge clk);

            expected_out = test_outputs[i];

            // Check output
            if (out !== expected_out) begin
                $display("FAIL: Input %b -> Output %b (expected %b)",
                         test_inputs[i], out, expected_out);
                errors = errors + 1;
            end
        end

        // Summary...
        $finish;
    end
endmodule
```

### Step 3: Simulate with Different Tools

#### Option A: Vivado Simulator (xsim)

```bash
# Compile the design
xvlog logic_net.v
xvlog tb_logic_net.v

# Elaborate
xelab -debug typical tb_logic_net -s tb_logic_net_sim

# Run simulation
xsim tb_logic_net_sim -runall

# View waveforms (GUI mode)
xsim tb_logic_net_sim -gui
```

#### Option B: Icarus Verilog

```bash
# Compile and run in one step
iverilog -o sim.out logic_net.v tb_logic_net.v
vvp sim.out

# View waveforms with GTKWave
gtkwave tb_logic_net.vcd
```

#### Option C: ModelSim/QuestaSim

```bash
# Create work library
vlib work

# Compile sources
vlog logic_net.v
vlog tb_logic_net.v

# Simulate
vsim -c tb_logic_net -do "run -all; quit"

# Or run with GUI
vsim tb_logic_net
# In ModelSim GUI: run -all
```

### Step 4: Analyze Results

#### Success Criteria
- All test vectors should produce matching outputs
- No 'X' or 'Z' values in outputs (indicates uninitialized or high-impedance states)
- Combinational delay should be minimal (typically < 1ns for simple gates)

#### Common Issues

**Mismatched outputs:**
- Verify test vector format (binary vs decimal)
- Check that Verilog module name matches instantiation in testbench
- Ensure input/output widths match

**X or Z values:**
- Usually indicates undriven wires
- Check all wires in generated Verilog have assignments

**Compilation errors:**
- Verify Verilog syntax
- Check for Verilog-1995 vs Verilog-2001 compatibility issues

### Self-Checking Testbench

For automated testing, create a self-checking testbench that exits with an error code:

```verilog
initial begin
    // ... run tests ...

    if (errors != 0) begin
        $display("FAIL: %0d errors detected", errors);
        $fatal(1, "Test failed");  // Exit with error
    end else begin
        $display("PASS: All tests passed");
    end
    $finish;
end
```

Then use in scripts:

```bash
#!/bin/bash
iverilog -o sim.out logic_net.v tb_logic_net.v && vvp sim.out
if [ $? -eq 0 ]; then
    echo "Simulation PASSED"
else
    echo "Simulation FAILED"
    exit 1
fi
```

---

## Synthesis with Vivado

Synthesis converts Verilog RTL code into gate-level netlists optimized for specific FPGA parts, providing resource estimates, timing analysis, and power consumption data.

### Overview

Synthesis provides:
- **Resource Estimates**: LUTs, FFs, DSPs, BRAM usage
- **Timing Analysis**: Maximum frequency (fmax), Worst Negative Slack (WNS), critical paths
- **Power Estimates**: Static and dynamic power consumption

### Prerequisites

- **Vivado Design Suite** installed (tested with 2019.1+)
- Generated Verilog file from TorchLogix (e.g., `logic_net.v`)
- Target FPGA part number (e.g., `xc7z020clg400-1` for Pynq-Z2)

### Quick Start

```bash
# Navigate to your Verilog output directory
cd verilog_output/

# Run synthesis with TCL script
vivado -mode batch -source synthesize.tcl -tclargs logic_net.v xc7z020clg400-1

# Results will be in synthesis_reports/
ls synthesis_reports/
# utilization.txt  timing.txt  power.txt
```

### Synthesis Methods

#### Method 1: Batch Mode with TCL Script (Recommended)

Create `synthesize.tcl`:

```tcl
# Parse arguments
set verilog_file [lindex $argv 0]
set part_number [lindex $argv 1]
set report_dir [lindex $argv 2]

if {$report_dir == ""} {
    set report_dir "synthesis_reports"
}

# Create output directory
file mkdir $report_dir

# Create in-memory project
create_project -in_memory -part $part_number

# Read Verilog source
read_verilog $verilog_file

# Detect if design has clock port
set has_clock [expr {[llength [get_ports -quiet clk]] > 0}]

if {$has_clock} {
    puts "INFO: Detected clocked design, applying timing constraints"

    # Create clock constraint (10ns period = 100 MHz)
    create_clock -period 10.0 -name clk [get_ports clk]

    # Input/output delays
    set_input_delay -clock clk 2.0 [get_ports -filter {NAME != clk && DIRECTION == IN}]
    set_output_delay -clock clk 2.0 [get_ports -filter {DIRECTION == OUT}]

    # Run synthesis
    synth_design -top [get_property TOP [current_fileset]]
} else {
    puts "INFO: Detected combinational design"

    # Run synthesis in out-of-context mode
    synth_design -top [get_property TOP [current_fileset]] -mode out_of_context
}

# Generate reports
report_utilization -file ${report_dir}/utilization.txt
report_timing_summary -file ${report_dir}/timing.txt
report_power -file ${report_dir}/power.txt

# Extract key metrics
set lut_count [get_property LUT_AS_LOGIC [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ LUT*}] | llength]
set ff_count [get_property PRIMITIVE_COUNT [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ REGISTER*}]]

puts "\n========================================"
puts "Synthesis Summary"
puts "========================================"
puts "LUTs:  $lut_count"
puts "FFs:   $ff_count"
puts "========================================"

# Save summary
set summary_file [open "${report_dir}/summary.txt" w]
puts $summary_file "LUTs: $lut_count"
puts $summary_file "FFs: $ff_count"
close $summary_file

puts "Reports saved to: $report_dir/"
exit
```

Run synthesis:

```bash
vivado -mode batch -source synthesize.tcl -tclargs logic_net.v xc7z020clg400-1 reports/
```

#### Method 2: Vivado GUI

1. Launch Vivado: `vivado`
2. Create New Project
   - Click "Create Project"
   - Choose project location
   - Select "RTL Project"
3. Add Verilog Source
   - In Flow Navigator: "Add Sources"
   - "Add or create design sources"
   - Add your `logic_net.v` file
4. Select Target Part
   - Common parts:
     - Pynq-Z2: `xc7z020clg400-1`
     - ZCU104: `xczu7ev-ffvc1156-2-e`
     - Artix-7: `xc7a35tcpg236-1`
5. Run Synthesis
   - In Flow Navigator: "Run Synthesis"
6. View Reports
   - After synthesis: "Open Synthesized Design"
   - Reports → Utilization, Timing Summary

#### Method 3: Vivado Tcl Console

```bash
vivado -mode tcl
```

Then in Tcl console:

```tcl
# Create in-memory project
create_project -in_memory -part xc7z020clg400-1

# Read Verilog
read_verilog logic_net.v

# Run synthesis
synth_design -top logic_net -mode out_of_context

# Generate reports
report_utilization -file utilization.txt
report_timing_summary -file timing.txt
report_power -file power.txt

exit
```

### Understanding the Reports

#### Utilization Report

Shows FPGA resource usage:

```
+-------------------------+------+-------+-----------+-------+
|        Site Type        | Used | Fixed | Available | Util% |
+-------------------------+------+-------+-----------+-------+
| Slice LUTs              |  147 |     0 |     53200 |  0.28 |
|   LUT as Logic          |  147 |     0 |     53200 |  0.28 |
|   LUT as Memory         |    0 |     0 |     17400 |  0.00 |
| Slice Registers         |    0 |     0 |    106400 |  0.00 |
|   Register as Flip Flop |    0 |     0 |    106400 |  0.00 |
|   Register as Latch     |    0 |     0 |    106400 |  0.00 |
| F7 Muxes                |    8 |     0 |     26600 |  0.03 |
| F8 Muxes                |    2 |     0 |     13300 |  0.02 |
+-------------------------+------+-------+-----------+-------+
```

**Key Metrics:**
- **Slice LUTs**: Primary logic resource. Each LUT can implement any 6-input Boolean function.
- **Slice Registers**: Flip-flops for sequential logic (0 for purely combinational designs)
- **F7/F8 Muxes**: Larger multiplexers for wide logic functions
- **DSPs**: Digital Signal Processing blocks (typically 0 for logic gate networks)
- **BRAM**: Block RAM (typically 0 for logic gate networks)

#### Timing Report

Shows timing analysis:

```
Timing Summary (ns)
-------------------
WNS(ns)      TNS(ns)      WHS(ns)      THS(ns)      WPWS(ns)     TPWS(ns)
-------      -------      -------      -------      --------     --------
  7.500        0.000        0.300        0.000         3.500        0.000
```

**Key Metrics:**
- **WNS (Worst Negative Slack)**: Most critical timing margin
  - Positive = timing met
  - Negative = timing violation
- **Critical Path Delay**: Longest combinational path through the design

**For combinational designs:**

```tcl
# Get maximum delay through the design
report_timing -delay_type min_max -max_paths 10 -file timing_paths.txt
```

Example output:
```
Critical Path Delay: 4.832 ns
Maximum Frequency: 206.95 MHz (if this were in a clocked design)
```

#### Power Report

Shows estimated power consumption:

```
Total On-Chip Power (W)  : 0.082
Dynamic (W)              : 0.012
Device Static (W)        : 0.070
```

### Interpreting Results for TorchLogix Models

#### Resource Estimates

**LUT Count Interpretation:**
- Each logic gate (AND, OR, XOR, etc.) typically maps to a fractional LUT
- Expect roughly 0.5-1 LUT per gate operation
- Tree structures may share LUTs efficiently
- A 1000-gate network might use 500-800 LUTs

**Resource Scaling:**
- Linear layers: O(neurons) LUTs
- Convolutional layers: O(kernels × receptive_field²) LUTs
- Deeper trees → more efficient LUT packing

**Pipelining Impact:**
- Adds flip-flops (FFs) for registers
- May actually **reduce** LUTs through better optimization
- Example: 50-layer model with 4 pipeline stages might use 20% fewer LUTs

#### Latency Estimates

**Combinational Delay:**
- Total latency = critical path delay
- Typical delays:
  - Simple AND/OR/XOR: 0.1-0.2 ns per gate
  - Deep trees (10 levels): 2-4 ns total
  - Very deep networks: 5-10 ns

**Maximum Frequency:**
```
fmax = 1 / critical_path_delay
```

Example: 4.5 ns critical path → fmax ≈ 222 MHz

### Pipelined vs Combinational Designs

#### Combinational (pipeline_stages=0)

```python
verilog = compiled.get_verilog_code(pipeline_stages=0)  # Default
```

- No clock or reset signals
- All logic in single cycle
- Synthesis uses `-mode out_of_context`
- **Problem:** May fail for large models (>1M Verilog lines)

**Timing Analysis:**
```
Combinational (pipeline_stages=0):
  Critical path: 25 ns
  fmax: 40 MHz
```

#### Pipelined (pipeline_stages>0)

```python
verilog = compiled.get_verilog_code(pipeline_stages=4)  # 4 stages
```

- Has clock (clk) and reset (rst) signals
- Logic divided into N pipeline stages
- Synthesis auto-detects and applies clock constraints
- **Solution:** Enables synthesis of very large models

**Timing Analysis:**
```
Pipelined (pipeline_stages=4):
  Critical path: 6 ns
  fmax: 166 MHz
  Latency: 4 cycles = 24 ns @ 166 MHz
```

**When to use pipelining:**
- Synthesis fails or runs for hours → Use `pipeline_stages=1` or more
- Model >20 layers → Consider `pipeline_stages=4-8`
- Model >100 layers → Use `pipeline_stages=16` or higher

### Synthesis Strategies

#### For Minimum Latency

```tcl
synth_design -top logic_net -mode out_of_context \
    -directive PerformanceOptimized \
    -no_lc  # Disable logic combining for speed
```

#### For Minimum Area

```tcl
synth_design -top logic_net -mode out_of_context \
    -directive AreaOptimized_high \
    -shreg_min_size 5  # Aggressive resource sharing
```

#### For Balanced Results

```tcl
synth_design -top logic_net -mode out_of_context \
    -directive Default
```

### Common Issues and Solutions

#### Issue: "Multi-driven net" Error
**Cause:** Multiple assign statements to the same wire.
**Solution:** Check generated Verilog for duplicate assignments.

#### Issue: Unrealistically High fmax
**Cause:** No input/output delays specified.
**Solution:** Add timing constraints with realistic I/O delays.

#### Issue: Very High LUT Count
**Cause:** Unoptimized gate structure or deep logic trees.
**Solution:**
- Check for constant propagation opportunities
- Verify gate operations are using optimal Boolean functions
- Consider factoring logic differently in training

### Next Steps After Synthesis

1. **Implementation**: Run place & route for final timing/resource numbers

```tcl
opt_design
place_design
route_design
report_timing_summary -file post_route_timing.txt
report_utilization -file post_route_utilization.txt
```

2. **Bitstream Generation**: Create FPGA configuration file

```tcl
write_bitstream -force logic_net.bit
```

3. **Hardware Testing**: Deploy to actual FPGA and validate

---

## Complete Workflow Examples

### Example 1: Small Model - Full Workflow

Train, export, test, and synthesize a small MNIST classifier:

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchlogix.layers import LogicDense, GroupSum
from torchlogix import CompiledLogicNet
import numpy as np
import subprocess

# 1. Create and train model
model = nn.Sequential(
    LogicDense(784, 128, connections="fixed", device="cpu"),
    LogicDense(128, 128, connections="fixed", device="cpu"),
    LogicDense(128, 100, connections="fixed", device="cpu"),
    GroupSum(10, tau=10.0)
)

# Train your model here...
# model.load_state_dict(torch.load('trained_model.pt'))

# 2. Compile and export Verilog
compiled = CompiledLogicNet(
    model,
    input_shape=(784,),
    use_bitpacking=False,
    num_bits=1
)

# Small model: fully combinational
verilog = compiled.get_verilog_code(
    module_name="mnist_classifier",
    pipeline_stages=0
)

with open('mnist_classifier.v', 'w') as f:
    f.write(verilog)

print("✓ Generated mnist_classifier.v")

# 3. Generate test vectors
num_tests = 100
test_inputs = np.random.randint(0, 2, (num_tests, 784), dtype=np.int8)
test_outputs = []

compiled.compile()
for inp in test_inputs:
    out = compiled.forward(inp.reshape(1, -1))
    test_outputs.append(out[0])

np.savetxt('test_vectors_input.txt', test_inputs, fmt='%d')
np.savetxt('test_vectors_output.txt', np.array(test_outputs), fmt='%d')

print(f"✓ Generated {num_tests} test vectors")

# 4. Run simulation (using Icarus Verilog)
subprocess.run([
    'iverilog', '-o', 'sim.out',
    'mnist_classifier.v', 'tb_mnist_classifier.v'
])
result = subprocess.run(['vvp', 'sim.out'], capture_output=True, text=True)

if "ALL TESTS PASSED" in result.stdout:
    print("✓ Simulation passed")
else:
    print("✗ Simulation failed")
    exit(1)

# 5. Run synthesis
subprocess.run([
    'vivado', '-mode', 'batch',
    '-source', 'synthesize.tcl',
    '-tclargs', 'mnist_classifier.v', 'xc7z020clg400-1', 'reports/'
])

print("✓ Synthesis complete - check reports/ directory")
```

### Example 2: Large Model with Pipelining

```python
#!/usr/bin/env python3
from torchlogix import CompiledLogicNet
import subprocess

# Load large pre-trained model
model = ...  # 50+ layers

compiled = CompiledLogicNet(model, input_shape=(784,), use_bitpacking=False, num_bits=1)

# Try different pipeline configurations
configs = [
    (0, "Fully combinational"),
    (1, "Single output register"),
    (4, "4 pipeline stages"),
    (8, "8 pipeline stages"),
]

for pipeline_stages, description in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {description} (pipeline_stages={pipeline_stages})")
    print(f"{'='*60}")

    # Generate Verilog
    verilog = compiled.get_verilog_code(
        module_name=f"large_model_p{pipeline_stages}",
        pipeline_stages=pipeline_stages
    )

    filename = f"large_model_p{pipeline_stages}.v"
    with open(filename, 'w') as f:
        f.write(verilog)

    print(f"✓ Generated {filename} ({len(verilog)} bytes)")

    # Synthesize
    result = subprocess.run([
        'vivado', '-mode', 'batch',
        '-source', 'synthesize.tcl',
        '-tclargs', filename, 'xc7z020clg400-1', f'reports_p{pipeline_stages}/'
    ], capture_output=True, text=True, timeout=600)

    if result.returncode == 0:
        print(f"✓ Synthesis succeeded")

        # Parse results
        with open(f'reports_p{pipeline_stages}/summary.txt') as f:
            print(f.read())
    else:
        print(f"✗ Synthesis failed or timed out")

print("\n" + "="*60)
print("Compare results in reports_p*/ directories")
print("="*60)
```

### Example 3: Convolutional Model

```python
#!/usr/bin/env python3
import torch.nn as nn
from torchlogix.layers import LogicConv2d, OrPooling, LogicDense, GroupSum
from torchlogix import CompiledLogicNet

# Create convolutional model
model = nn.Sequential(
    LogicConv2d(
        in_dim=(28, 28),
        channels=1,
        num_kernels=16,
        tree_depth=3,
        receptive_field_size=5,
        padding=2,
        connections="fixed",
        device="cpu"
    ),
    OrPooling(kernel_size=2, stride=2),

    LogicConv2d(
        in_dim=(14, 14),
        channels=16,
        num_kernels=32,
        tree_depth=3,
        receptive_field_size=3,
        padding=1,
        connections="fixed",
        device="cpu"
    ),
    OrPooling(kernel_size=2, stride=2),

    nn.Flatten(),
    LogicDense(32*7*7, 256, connections="fixed", device="cpu"),
    LogicDense(256, 100, connections="fixed", device="cpu"),
    GroupSum(10, tau=10.0)
)

# Train model...

# Export with pipelining (medium-sized model)
compiled = CompiledLogicNet(
    model,
    input_shape=(1, 28, 28),
    use_bitpacking=False,
    num_bits=1
)

verilog = compiled.get_verilog_code(
    module_name="conv_classifier",
    pipeline_stages=4  # 4 stages for medium conv model
)

compiled.export_hdl(
    "./conv_verilog_output",
    module_name="conv_classifier",
    pipeline_stages=4
)

print("✓ Exported convolutional model to conv_verilog_output/")
```

### Example 4: Python-Driven Synthesis Loop

Automate synthesis and collect metrics:

```python
#!/usr/bin/env python3
import subprocess
import re
import pandas as pd

def synthesize_and_extract_metrics(verilog_file, part, report_dir):
    """Run Vivado synthesis and extract key metrics."""

    # Run synthesis
    result = subprocess.run([
        'vivado', '-mode', 'batch',
        '-source', 'synthesize.tcl',
        '-tclargs', verilog_file, part, report_dir
    ], capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        return None

    # Parse summary
    metrics = {}
    with open(f'{report_dir}/summary.txt', 'r') as f:
        for line in f:
            if 'LUTs:' in line:
                metrics['luts'] = int(re.search(r'\d+', line).group())
            elif 'FFs:' in line:
                metrics['ffs'] = int(re.search(r'\d+', line).group())

    # Parse timing
    with open(f'{report_dir}/timing.txt', 'r') as f:
        content = f.read()
        wns_match = re.search(r'WNS\(ns\)\s+([-\d.]+)', content)
        if wns_match:
            metrics['wns_ns'] = float(wns_match.group(1))

    return metrics

# Run experiments
compiled = CompiledLogicNet(model, input_shape=(784,), use_bitpacking=False, num_bits=1)

results = []
for stages in [0, 1, 2, 4, 8, 16]:
    print(f"Synthesizing with pipeline_stages={stages}...")

    verilog = compiled.get_verilog_code(
        module_name=f'model_p{stages}',
        pipeline_stages=stages
    )

    filename = f'model_p{stages}.v'
    with open(filename, 'w') as f:
        f.write(verilog)

    metrics = synthesize_and_extract_metrics(
        filename,
        'xc7z020clg400-1',
        f'reports_p{stages}'
    )

    if metrics:
        metrics['pipeline_stages'] = stages
        results.append(metrics)
        print(f"  LUTs: {metrics['luts']}, FFs: {metrics['ffs']}, WNS: {metrics['wns_ns']} ns")

# Create summary DataFrame
df = pd.DataFrame(results)
df.to_csv('synthesis_comparison.csv', index=False)

print("\n" + "="*60)
print("Synthesis Comparison")
print("="*60)
print(df)
```

---

## Advanced Topics

### Custom Timing Constraints

For more accurate timing analysis, create custom constraints in `constraints.xdc`:

```tcl
# Virtual clock for timing analysis (10ns = 100 MHz)
create_clock -period 10.000 -name virtual_clk

# Input delay (assume inputs arrive 2ns after clock edge)
set_input_delay -clock virtual_clk 2.000 [get_ports inp*]

# Output delay (assume outputs must be stable 2ns before next clock edge)
set_output_delay -clock virtual_clk 2.000 [get_ports out*]

# For pipelined designs with actual clock
create_clock -period 5.000 -name clk [get_ports clk]  # 200 MHz

# Relax timing on reset path
set_false_path -from [get_ports rst]
```

Load in synthesis:

```tcl
read_xdc constraints.xdc
synth_design -top logic_net
```

### Comparing C Code and Verilog Implementations

TorchLogix can generate both C code (for HLS or CPU) and Verilog:

```python
from torchlogix import CompiledLogicNet

compiled = CompiledLogicNet(model, input_shape=(784,), use_bitpacking=False, num_bits=1)

# Generate C code
c_code = compiled.get_c_code()
with open('model.c', 'w') as f:
    f.write(c_code)

# Generate Verilog
verilog_code = compiled.get_verilog_code()
with open('model.v', 'w') as f:
    f.write(verilog_code)

# Compile C for CPU execution
compiled.compile(compiler='gcc', optimization_level='-O3')

# Now you can compare:
# - C compiled to native CPU code
# - C compiled through Vivado HLS to RTL
# - Direct Verilog synthesis

# All three should produce functionally identical results
```

### Integration with C→HLS Pipeline

For users who want to explore the HLS route:

```python
# Generate optimized C code
compiled = CompiledLogicNet(model)
c_code = compiled.get_c_code()

with open('model.c', 'w') as f:
    f.write(c_code)
```

Then use Vivado HLS:

```tcl
# hls_script.tcl
open_project hls_project
set_top model_top
add_files model.c
open_solution "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10

csynth_design
export_design -format ip_catalog
exit
```

```bash
vivado_hls -f hls_script.tcl
```

This generates an IP core that can be integrated into larger FPGA designs.

### Layer Support Status and Future Work

#### Currently Supported

| Layer | Status | Implementation |
|-------|--------|---------------|
| LogicDense | ✅ Complete | Direct gate synthesis |
| LogicConv2d | ✅ Complete | Binary tree structure |
| LogicConv3d | ✅ Complete | Binary tree with 3D indexing |
| Flatten | ✅ Complete | Wire passthrough |

#### In Progress / TODO

| Layer | Status | Notes |
|-------|--------|-------|
| OrPooling | ⚠️ TODO | OR reduction tree needed |
| GroupSum | ⚠️ TODO | Adder tree implementation needed |

Models using unsupported layers will generate placeholders in Verilog. For production use:
- Use C code generation for complete model support
- Wait for future releases with complete Verilog support
- Manually implement missing layers if needed

### Optimization Tips

**Model Architecture:**
- Prefer models with 20-100 layers for good synthesis results
- Very deep models (>200 layers) require aggressive pipelining
- Conv layers with large receptive fields may need optimization

**Pipeline Configuration:**
- Start with `pipeline_stages=0` for small models
- Increase incrementally if synthesis fails or is slow
- Use full pipelining (`pipeline_stages=999`) for maximum fmax

**Synthesis Directives:**
- Use `-directive PerformanceOptimized` for speed
- Use `-directive AreaOptimized_high` for small FPGAs
- Experiment with different strategies for your specific design

**Resource Utilization:**
- Target <70% LUT utilization for good place-and-route results
- Very high utilization (>90%) may cause routing failures
- Consider model size vs available FPGA resources during training

---

## Summary

TorchLogix provides comprehensive support for deploying logic gate networks to FPGA hardware:

1. **Direct Verilog Generation**: Export trained models to readable, synthesizable Verilog RTL
2. **Configurable Pipelining**: Balance latency and synthesis complexity with flexible pipeline stages
3. **Testing Support**: Generate test vectors and verify functionality with standard simulators
4. **Synthesis Integration**: Automate synthesis with Vivado to obtain resource and timing estimates
5. **Complete Workflows**: End-to-end examples from training to hardware deployment

**Key Takeaways:**
- Use `pipeline_stages=0` for small models, increase for larger ones
- Always test generated Verilog with simulations before synthesis
- Synthesis provides accurate resource and timing estimates before hardware deployment
- Both Verilog and C code generation are supported for maximum flexibility

For questions or issues with hardware deployment, please refer to the [TorchLogix repository](https://github.com/your-org/torchlogix) or open an issue.
