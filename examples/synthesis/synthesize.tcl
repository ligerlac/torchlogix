#!/usr/bin/env vivado -mode batch -source
# synthesize.tcl - Automated Vivado synthesis script for TorchLogix Verilog
#
# Usage:
#   vivado -mode batch -source synthesize.tcl -tclargs <verilog_file> <part> [output_dir] [clock_period_ns]
#
# Arguments:
#   verilog_file    - Path to Verilog file to synthesize
#   part            - Target FPGA part (e.g., xc7z020clg400-1)
#   output_dir      - Optional: Directory for reports (default: synthesis_reports)
#   clock_period_ns - Optional: Target clock period in ns for timing analysis (default: 10.0)
#
# Examples:
#   vivado -mode batch -source synthesize.tcl -tclargs logic_net.v xc7z020clg400-1
#   vivado -mode batch -source synthesize.tcl -tclargs logic_net.v xczu7ev-ffvc1156-2-e my_reports/
#   vivado -mode batch -source synthesize.tcl -tclargs logic_net.v xc7z020clg400-1 reports/ 5.0

# Parse command-line arguments
if { $argc < 2 } {
    puts "ERROR: Insufficient arguments"
    puts "Usage: synthesize.tcl <verilog_file> <part> \[output_dir\] \[clock_period_ns\]"
    exit 1
}

set verilog_file [lindex $argv 0]
set part [lindex $argv 1]

# Optional arguments with defaults
if { $argc >= 3 } {
    set output_dir [lindex $argv 2]
} else {
    set output_dir "synthesis_reports"
}

if { $argc >= 4 } {
    set clock_period [lindex $argv 3]
} else {
    set clock_period 10.0
}

# Validate inputs
if { ![file exists $verilog_file] } {
    puts "ERROR: Verilog file not found: $verilog_file"
    exit 1
}

# Create output directory
file mkdir $output_dir

puts "========================================="
puts "TorchLogix Verilog Synthesis"
puts "========================================="
puts "Verilog file:    $verilog_file"
puts "Target part:     $part"
puts "Output dir:      $output_dir"
puts "Clock period:    ${clock_period} ns"
puts "========================================="
puts ""

# Extract module name from filename (remove path and .v extension)
set module_name [file rootname [file tail $verilog_file]]
puts "Module name:     $module_name"
puts ""

# Create in-memory project
puts "Creating in-memory project..."
create_project -in_memory -part $part

# Read Verilog source
puts "Reading Verilog source..."
read_verilog $verilog_file

# Detect if design is clocked (has clk port)
set has_clock 0
if { [catch {
    set file_contents [read [open $verilog_file r]]
    if { [regexp {input\s+(wire\s+)?clk} $file_contents] } {
        set has_clock 1
    }
}] } {
    # If file read fails, assume combinational
    set has_clock 0
}

if { $has_clock } {
    puts "Detected clocked design (has clk port)"
} else {
    puts "Detected combinational design (no clk port)"
}

# Run synthesis
puts ""
puts "Running synthesis..."
puts "This may take 1-5 minutes depending on design size..."
set start_time [clock seconds]

# Run synthesis with appropriate strategy
if { $has_clock } {
    # Clocked design - use normal synthesis and apply real clock constraints BEFORE synthesis
    puts "Applying clock constraints before synthesis..."
    create_clock -period $clock_period -name clk [get_ports clk]
    set_input_delay -clock clk [expr $clock_period * 0.2] [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst}]
    set_output_delay -clock clk [expr $clock_period * 0.2] [get_ports -filter {DIRECTION == OUT}]

    if { [catch {synth_design -top $module_name -directive Default} result] } {
        puts "ERROR: Synthesis failed"
        puts $result
        exit 1
    }
} else {
    # Combinational design - use out_of_context mode
    if { [catch {synth_design -top $module_name -mode out_of_context -directive Default} result] } {
        puts "ERROR: Synthesis failed"
        puts $result
        exit 1
    }

    # Create virtual timing constraints AFTER synthesis for combinational logic analysis
    puts "Creating virtual timing constraints for analysis..."
    create_clock -period $clock_period -name virtual_clk
    set_input_delay -clock virtual_clk [expr $clock_period * 0.2] [get_ports -filter {DIRECTION == IN}]
    set_output_delay -clock virtual_clk [expr $clock_period * 0.2] [get_ports -filter {DIRECTION == OUT}]
}

set end_time [clock seconds]
set synthesis_time [expr $end_time - $start_time]
puts "Synthesis completed in ${synthesis_time} seconds"
puts ""

# Generate reports
puts "Generating reports..."

# Utilization report
puts "  - Utilization report: ${output_dir}/utilization.txt"
report_utilization -file ${output_dir}/utilization.txt

# Detailed utilization with hierarchical breakdown
puts "  - Hierarchical utilization: ${output_dir}/utilization_hierarchical.txt"
report_utilization -hierarchical -file ${output_dir}/utilization_hierarchical.txt

# Timing summary
puts "  - Timing summary: ${output_dir}/timing_summary.txt"
report_timing_summary -file ${output_dir}/timing_summary.txt

# Detailed timing paths
puts "  - Detailed timing paths: ${output_dir}/timing_paths.txt"
report_timing -delay_type min_max -max_paths 10 -sort_by slack -file ${output_dir}/timing_paths.txt

# Power report
puts "  - Power report: ${output_dir}/power.txt"
report_power -file ${output_dir}/power.txt

# Design analysis
puts "  - Design analysis: ${output_dir}/design_analysis.txt"
report_design_analysis -file ${output_dir}/design_analysis.txt

puts ""
puts "Extracting key metrics..."

# Extract key metrics from reports
set summary_file [open ${output_dir}/summary.txt w]
puts $summary_file "========================================="
puts $summary_file "Synthesis Summary"
puts $summary_file "========================================="
puts $summary_file "Design:          $module_name"
puts $summary_file "Target part:     $part"
puts $summary_file "Clock period:    ${clock_period} ns"
puts $summary_file "Synthesis time:  ${synthesis_time} seconds"
puts $summary_file ""

# Parse utilization
set util_file [open ${output_dir}/utilization.txt r]
set util_content [read $util_file]
close $util_file

# Extract LUT count
if { [regexp {Slice LUTs\s+\|\s+(\d+)} $util_content match lut_count] } {
    puts $summary_file "LUTs:            $lut_count"
    puts "LUTs used: $lut_count"
} else {
    puts $summary_file "LUTs:            N/A"
}

# Extract FF count
if { [regexp {Slice Registers\s+\|\s+(\d+)} $util_content match ff_count] } {
    puts $summary_file "Flip-Flops:      $ff_count"
    puts "Flip-Flops used: $ff_count"
} else {
    puts $summary_file "Flip-Flops:      N/A"
}

# Extract DSP count
if { [regexp {DSPs\s+\|\s+(\d+)} $util_content match dsp_count] } {
    puts $summary_file "DSPs:            $dsp_count"
} else {
    puts $summary_file "DSPs:            0"
}

# Extract BRAM count
if { [regexp {Block RAM Tile\s+\|\s+(\d+)} $util_content match bram_count] } {
    puts $summary_file "BRAM Tiles:      $bram_count"
} else {
    puts $summary_file "BRAM Tiles:      0"
}

puts $summary_file ""

# Parse timing
set timing_file [open ${output_dir}/timing_summary.txt r]
set timing_content [read $timing_file]
close $timing_file

# Extract WNS
if { [regexp {WNS\(ns\)\s+([-\d.]+)} $timing_content match wns] } {
    puts $summary_file "WNS:             ${wns} ns"
    puts "WNS: ${wns} ns"

    # Calculate achieved frequency
    if { $wns >= 0 } {
        set achieved_period [expr $clock_period - $wns]
        set fmax [expr 1000.0 / $achieved_period]
        puts $summary_file "Achieved fmax:   [format %.2f $fmax] MHz (based on ${clock_period}ns target)"
        puts "Achieved fmax: [format %.2f $fmax] MHz"
    } else {
        puts $summary_file "Achieved fmax:   Failed to meet timing"
        puts "WARNING: Failed to meet timing constraint"
    }
} else {
    puts $summary_file "WNS:             N/A"
}

# Extract critical path delay from detailed timing
set timing_paths_file [open ${output_dir}/timing_paths.txt r]
set timing_paths_content [read $timing_paths_file]
close $timing_paths_file

if { [regexp {data path delay:\s+([\d.]+)ns} $timing_paths_content match delay] } {
    puts $summary_file "Critical path:   ${delay} ns"
    puts "Critical path delay: ${delay} ns"

    # This is the actual combinational latency
    puts $summary_file "Latency:         ${delay} ns (1 clock cycle)"
} else {
    puts $summary_file "Critical path:   N/A"
}

puts $summary_file ""

# Parse power
set power_file [open ${output_dir}/power.txt r]
set power_content [read $power_file]
close $power_file

if { [regexp {Total On-Chip Power \(W\)\s+:\s+([\d.]+)} $power_content match total_power] } {
    puts $summary_file "Total power:     ${total_power} W"
}

if { [regexp {Dynamic \(W\)\s+:\s+([\d.]+)} $power_content match dynamic_power] } {
    puts $summary_file "Dynamic power:   ${dynamic_power} W"
}

if { [regexp {Device Static \(W\)\s+:\s+([\d.]+)} $power_content match static_power] } {
    puts $summary_file "Static power:    ${static_power} W"
}

puts $summary_file ""
puts $summary_file "========================================="
close $summary_file

puts ""
puts "========================================="
puts "Synthesis Complete!"
puts "========================================="
puts "Reports available in: $output_dir"
puts ""
puts "Key files:"
puts "  - summary.txt               - Quick summary of key metrics"
puts "  - utilization.txt           - Resource utilization"
puts "  - timing_summary.txt        - Timing analysis"
puts "  - timing_paths.txt          - Critical path details"
puts "  - power.txt                 - Power estimates"
puts ""
puts "View summary:"
puts "  cat ${output_dir}/summary.txt"
puts "========================================="

# Optional: Save checkpoint for further analysis
# write_checkpoint -force ${output_dir}/post_synth.dcp

exit 0
