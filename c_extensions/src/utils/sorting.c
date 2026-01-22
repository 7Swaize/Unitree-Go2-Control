#include "sorting.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>



void build_histogram(uint64_t* keys, Py_ssize_t N, int shift, Py_ssize_t* hist) {
    memset(hist, 0, RADIX * sizeof(int));
    for (Py_ssize_t i = 0; i < N; i++) {
        uint8_t d = msb_digit(keys[i], shift);
        hist[d]++;
    }
}

void compute_heads_tails(Py_ssize_t* hist, Py_ssize_t* heads, Py_ssize_t* tails) {
    heads[0] = 0;
    tails[0] = hist[0];

    for (int i = 1; i < RADIX; i++) {
        heads[i] = heads[i - 1] + hist[i - 1];
        tails[i] = heads[i] + hist[i];
    }
}

void permute(uint64_t* keys, Py_ssize_t* indices, int shift, int passes, Py_ssize_t* heads, Py_ssize_t* tails) {
    
}

void radix_64_inp_par(uint64_t* keys, Py_ssize_t* indices, Py_ssize_t N, int shift, int passes) {
    if (N <= 1 || shift >= passes) {
        return;
    }

    Py_ssize_t hist[RADIX], heads[RADIX], tails[RADIX];

    build_histogram(keys, N, shift, hist);
    compute_heads_tails(hist, heads, tails);
    
}

