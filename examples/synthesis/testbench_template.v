`timescale 1ns/1ps

// Testbench template for TorchLogix-generated Verilog modules
//
// Instructions:
// 1. Adjust INPUT_WIDTH and OUTPUT_WIDTH to match your module
// 2. Update the module name in the instantiation (replace 'logic_net')
// 3. Generate test vectors using Python (see generate_test_vectors.py)
// 4. Run simulation:
//    iverilog -o sim.out your_module.v testbench_template.v && vvp sim.out
//    OR
//    xvlog your_module.v testbench_template.v && xelab tb_logic_net && xsim tb_logic_net -runall

module tb_logic_net;
    // =================================================================
    // Configuration Parameters - ADJUST THESE FOR YOUR MODULE
    // =================================================================
    parameter INPUT_WIDTH = 8;          // Input bus width
    parameter OUTPUT_WIDTH = 2;         // Output bus width
    parameter NUM_TESTS = 100;          // Number of test vectors
    parameter CLOCK_PERIOD = 10;        // Clock period in ns (for timing reference)

    // =================================================================
    // Signals
    // =================================================================
    reg [INPUT_WIDTH-1:0] inp;          // Input to DUT
    wire [OUTPUT_WIDTH-1:0] out;        // Output from DUT
    reg [OUTPUT_WIDTH-1:0] expected_out; // Expected output

    // Test vectors storage
    reg [INPUT_WIDTH-1:0] test_inputs [0:NUM_TESTS-1];
    reg [OUTPUT_WIDTH-1:0] test_outputs [0:NUM_TESTS-1];

    // Test control
    integer i;
    integer errors;
    integer test_num;
    real start_time, end_time;

    // =================================================================
    // DUT Instantiation - CHANGE MODULE NAME HERE
    // =================================================================
    logic_net dut (
        .inp(inp),
        .out(out)
    );

    // =================================================================
    // Test Vector Loading
    // =================================================================
    initial begin
        // Load test vectors from files
        // Generate these files using the Python script
        if ($test$plusargs("vectors")) begin
            $value$plusargs("vectors=%s", test_vector_dir);
            $readmemb({test_vector_dir, "/test_vectors_input.txt"}, test_inputs);
            $readmemb({test_vector_dir, "/test_vectors_output.txt"}, test_outputs);
        end else begin
            // Default location
            $readmemb("test_vectors_input.txt", test_inputs);
            $readmemb("test_vectors_output.txt", test_outputs);
        end

        errors = 0;
        test_num = 0;
    end

    // =================================================================
    // Test Stimulus
    // =================================================================
    initial begin
        // Initialize
        inp = 0;
        #1;  // Wait for initialization

        // Print header
        $display("\n========================================");
        $display("TorchLogix Verilog Testbench");
        $display("========================================");
        $display("Module:      logic_net");
        $display("Input bits:  %0d", INPUT_WIDTH);
        $display("Output bits: %0d", OUTPUT_WIDTH);
        $display("Test count:  %0d", NUM_TESTS);
        $display("========================================\n");

        $display("Time(ns)\tTest#\tInput\t\t\tOutput\t\tExpected\tStatus");
        $display("-------\t\t-----\t-----\t\t\t------\t\t--------\t------");

        start_time = $realtime;

        // Run through all test vectors
        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            test_num = i;
            inp = test_inputs[i];
            expected_out = test_outputs[i];

            // Wait for combinational logic to settle
            #CLOCK_PERIOD;

            // Check output
            if (out !== expected_out) begin
                $display("%0t\t%0d\t%b\t%b\t%b\t\tFAIL",
                         $time, test_num, inp, out, expected_out);
                errors = errors + 1;

                // Optional: Stop on first error (comment out for full test)
                // $display("\nStopping on first error.");
                // $finish;
            end else begin
                // Only print passing tests in verbose mode
                if ($test$plusargs("verbose")) begin
                    $display("%0t\t%0d\t%b\t%b\t%b\t\tPASS",
                             $time, test_num, inp, out, expected_out);
                end
            end
        end

        end_time = $realtime;

        // Print summary
        print_summary();

        // Exit with appropriate code
        if (errors == 0) begin
            $finish;
        end else begin
            $fatal(1, "Test failed with %0d errors", errors);
        end
    end

    // =================================================================
    // Summary Reporting
    // =================================================================
    task print_summary;
        real sim_time;
        real throughput;
        begin
            sim_time = end_time - start_time;
            throughput = NUM_TESTS / (sim_time / 1000.0);  // Tests per us

            $display("\n========================================");
            $display("Test Summary");
            $display("========================================");
            $display("Total tests:     %0d", NUM_TESTS);
            $display("Passed:          %0d", NUM_TESTS - errors);
            $display("Failed:          %0d", errors);
            $display("Success rate:    %0.2f%%", (NUM_TESTS - errors) * 100.0 / NUM_TESTS);
            $display("Simulation time: %0.2f ns", sim_time);
            $display("Throughput:      %0.2f tests/us", throughput);
            $display("========================================");

            if (errors == 0) begin
                $display("RESULT: ALL TESTS PASSED");
            end else begin
                $display("RESULT: TESTS FAILED");
            end
            $display("========================================\n");
        end
    endtask

    // =================================================================
    // Waveform Dumping (optional)
    // =================================================================
    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("tb_logic_net.vcd");
            $dumpvars(0, tb_logic_net);
            $display("VCD waveform dumping enabled: tb_logic_net.vcd");
        end
    end

    // =================================================================
    // Timeout Watchdog
    // =================================================================
    initial begin
        #(CLOCK_PERIOD * NUM_TESTS * 2);  // 2x expected time
        $display("\nERROR: Simulation timeout!");
        $fatal(1, "Watchdog timeout");
    end

    // =================================================================
    // Assertion Checks (optional - enable with +define+ENABLE_ASSERTIONS)
    // =================================================================
    `ifdef ENABLE_ASSERTIONS
    // Check for X or Z values in output
    always @(out) begin
        if (^out === 1'bx) begin
            $error("Time %0t: Output contains X values: %b", $time, out);
        end
    end

    // Check for stable outputs
    property output_stable;
        @(inp) ##1 (out == $past(out) || inp != $past(inp));
    endproperty

    assert property (output_stable)
        else $error("Output changed without input change at time %0t", $time);
    `endif

endmodule

// =================================================================
// Alternative: Randomized Testing Module
// =================================================================
module tb_logic_net_random;
    parameter INPUT_WIDTH = 8;
    parameter OUTPUT_WIDTH = 2;
    parameter NUM_RANDOM_TESTS = 1000;
    parameter CLOCK_PERIOD = 10;

    reg [INPUT_WIDTH-1:0] inp;
    wire [OUTPUT_WIDTH-1:0] out;
    wire [OUTPUT_WIDTH-1:0] out_reference;

    integer i;

    // DUT
    logic_net dut (
        .inp(inp),
        .out(out)
    );

    // Reference model (if available)
    // logic_net_reference ref (
    //     .inp(inp),
    //     .out(out_reference)
    // );

    initial begin
        $display("Random testing with %0d vectors", NUM_RANDOM_TESTS);

        for (i = 0; i < NUM_RANDOM_TESTS; i = i + 1) begin
            inp = $random;
            #CLOCK_PERIOD;

            // Check against reference
            // if (out !== out_reference) begin
            //     $display("Mismatch at test %0d: inp=%b out=%b ref=%b",
            //              i, inp, out, out_reference);
            // end

            // Check for X/Z
            if (^out === 1'bx) begin
                $error("X/Z detected in output at test %0d", i);
            end
        end

        $display("Random testing complete");
        $finish;
    end
endmodule
