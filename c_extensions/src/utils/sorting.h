#ifndef SORTING_H
#define SORTING_H

#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <Python.h>
#include <arm_neon.h>

#define RADIX 256
#define PASSES 8


void radix_sort_64(uint64_t* keys, Py_ssize_t* indices,  Py_ssize_t N);


void radix_64_inp_par(uint64_t* keys, Py_ssize_t* indices, Py_ssize_t N, int pass);

void build_histogram(const uint64_t* keys, Py_ssize_t N, int pass, Py_ssize_t* hist);

void compute_heads_tails(const Py_ssize_t* hist, Py_ssize_t* heads, Py_ssize_t* tails);

void permute(uint64_t* keys, Py_ssize_t* indices, int pass, const Py_ssize_t* heads, const Py_ssize_t* tails, const Py_ssize_t* hist);

void repair(uint64_t* keys, Py_ssize_t* indices, int pass, const Py_ssize_t* heads, const Py_ssize_t* tails);

static inline uint8_t msb(uint64_t num, int pass) {
    return (num >> (8 * (7 - pass))) & 0xFF;
}


#endif