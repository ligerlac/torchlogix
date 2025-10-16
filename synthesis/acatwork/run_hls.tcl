# Create a project
open_project -reset acat_model_project

# Add design files
add_files acat_model.cpp
add_files acat_model.h
add_files x_val.txt
add_files y_val_ref.txt
# Add test bench & files
add_files -tb acat_model_tb.cpp

# Set the top-level function
set_top apply_logic_net_one_event

# ########################################################
# Create a solution
open_solution -reset solution1
# Define technology and clock rate
set_part  virtex7
create_clock -period 6.25 -name default

# Source x_hls.tcl to determine which steps to execute
source x_hls.tcl
#csim_design

# Set any optimization directives
# End of directives

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	csynth_design
	
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	
	cosim_design
} elseif {$hls_exec == 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	
	cosim_design
	export_design -format ip_catalog
} 

exit
