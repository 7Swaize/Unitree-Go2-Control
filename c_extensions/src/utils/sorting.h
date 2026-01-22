#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <Python.h>

#define RADIX 256
#define PASSES 8
#define THREADS 8

// stanford paper: https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/he.pdf
// more math heavy paper: https://scispace.com/pdf/paradis-an-efficient-parallel-algorithm-for-in-place-radix-4zo3nn8gi4.pdf


void radix_64_inp_par(uint64_t* keys, Py_ssize_t* indices, Py_ssize_t N, int shift, int passes);

void build_histogram(uint64_t* keys, Py_ssize_t N, int shift, Py_ssize_t* hist);

void compute_heads_tails(Py_ssize_t* hist, Py_ssize_t* heads, Py_ssize_t* tails);

void permute(uint64_t* keys, Py_ssize_t* indices, int shift, int passes, Py_ssize_t* heads, Py_ssize_t* tails);

static inline uint8_t msb_digit(uint64_t num, int pass) {
    return (num >> (8 * (7 - pass))) & 0xFF;
}


#endif