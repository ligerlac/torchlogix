// types.h
#ifndef TYPES_H
#define TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>

// 10-bit input type
typedef ap_uint<10> in_t;

// 16-bit fixed point output type (8 int, 8 frac)
typedef ap_ufixed<16, 8> out_t;

// 1-bit intermediate bool type
typedef ap_uint<1> bool_t;

//type for encoding which logic gate
typedef ap_uint<4> gate_t;

//type for indices in conv
typedef ap_uint<5> index_t;

#endif