#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <Python.h>
#include <immintrin.h>


#define RADIX 256
#define PASSES 8


void radix_sort_64(uint64_t* keys, Py_ssize_t* indices,  Py_ssize_t N);


void radix_64_inp_par(uint64_t* keys, Py_ssize_t* indices, Py_ssize_t N, int pass);

void build_histogram(uint64_t* keys, Py_ssize_t N, int shift, Py_ssize_t* hist);

void compute_heads_tails(Py_ssize_t* hist, Py_ssize_t* heads, Py_ssize_t* tails);

void permute(uint64_t* keys, Py_ssize_t* indices, int pass, Py_ssize_t* heads, Py_ssize_t* tails, Py_ssize_t* hist);

void repair(uint64_t* keys, Py_ssize_t* indices, int pass, Py_ssize_t* heads, Py_ssize_t* tails);

static inline uint8_t msb(uint64_t num, int pass) {
    return (num >> (8 * (7 - pass))) & 0xFF;
}


#endif