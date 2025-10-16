#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "acat_model.h"
#include "types.h"

// Function to read input from a file
in_t *read_input_from_file(const char *filename, size_t n_events) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s for reading.\n", filename);
        return NULL;
    }

    size_t capacity = n_events * 18 * 14;
    in_t *input = (in_t *)malloc(capacity * sizeof(in_t));
    if (!input) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }
    
    size_t line = 0;
    size_t index = 0;
    in_t val;
    size_t vals_in_line = 0;

    while (fscanf(file, "%d", &val) == 1) {
        if (val < 0 || val > 256) {
            fprintf(stderr, "Invalid value: %d at index %zu\n", val, index);
            free(input);
            fclose(file);
            return NULL;
        }

        input[index++] = val;
        vals_in_line++;

        if (vals_in_line == 18 * 14) {
            vals_in_line = 0;
            line++;

            if (line >= n_events) break;
        }
    }

    if (line < n_events) {
        printf("Warning: requested %zu lines, but only %zu available in file.\n", n_events, line);
    }

    fclose(file);
    return input;

}

// Function to read exactly n_events lines from reference file
out_t *read_expected_output_from_file(const char *filename, size_t n_events) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    out_t *output = (out_t *)malloc(n_events * sizeof(out_t));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    size_t index = 0;
    while (index < n_events && fscanf(file, "%d", &output[index]) == 1) {
        index++;
    }

    if (index < n_events) {
        fprintf(stderr, "Warning: Expected %zu values, but only read %zu from file.\n", n_events, index);
    }

    fclose(file);

    return output;
}


void apply_logic_net(in_t const *inp, out_t *out, size_t len) {
    in_t *inp_temp = (in_t *)malloc(252*sizeof(in_t));
    out_t *out_temp = (out_t *)malloc(1*sizeof(out_t));

    // run inference one event at a time
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < 252; ++j) {
            inp_temp[j] = inp[i * 252 + j];
        }
        apply_logic_net_one_event(inp_temp, out_temp);
        for (size_t k = 0; k < 1; ++k) {
            out[i * 1 + k] = out_temp[k];
        }
    }
    free(inp_temp);
    free(out_temp);
}


int main() {
    size_t n_events = 20;
    printf("n_events = %ld\n", n_events);

    in_t *test_input = read_input_from_file("x_val.txt", n_events);
    out_t *test_output = (out_t *)malloc(n_events * sizeof(out_t));

    apply_logic_net(test_input, test_output, n_events);

    printf("test_output = \n");
    for (size_t j = 0; j < n_events; ++j) { 
        printf("%d ", test_output[j]);
        if ((j + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    out_t *expected_output = read_expected_output_from_file("y_val_ref.txt", n_events);
    printf("expected_output = \n");
    for (size_t k = 0; k < n_events; ++k) { 
        printf("%d ", expected_output[k]);
        if ((k + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    int match = 1;
    for (size_t l = 0; l < n_events; ++l) {
        if (test_output[l] != expected_output[l]) {
            fprintf(stderr, "Mismatch at index %zu: expected %d, got %d\n", l, expected_output[l], test_output[l]);
            match = 0;
        }
    }

    printf(match ? "Test passed!\n" : "Test failed!\n");

    free(test_input);
    free(test_output);
    free(expected_output);

    return match;
}
