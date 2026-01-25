#include "sorting.h"

/*
void build_histogram(uint64_t* keys, Py_ssize_t N, int pass, Py_ssize_t* hist) {
    memset(hist, 0, RADIX * sizeof(Py_ssize_t));


    #pragma omp parallel
    {
        Py_ssize_t local[RADIX] = {0};

        #pragma omp for schedule(static)
        for (Py_ssize_t i = 0; i < N; i++) {
            local[msb(keys[i], pass)]++;

        }

        #pragma omp critical
        {
            // gloval merge optmize SIMD later
            for (int b = 0; b < RADIX; b++) {
                hist[b] += local[b];
            }
        }
    }
}
*/


// https://stackoverflow.com/questions/16789242/fill-histograms-array-reduction-in-parallel-with-openmp-without-using-a-critic
void build_histogram(uint64_t* keys, Py_ssize_t N, int pass, Py_ssize_t* hist) {
    Py_ssize_t* thist;

    #pragma omp parallel 
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        #pragma omp single
        thist = calloc(nthreads * RADIX, sizeof(Py_ssize_t));

        Py_ssize_t* local = &thist[tid * RADIX];

        #pragma omp for schedule(static)
        for (Py_ssize_t i = 0; i < N; i++) {
            uint8_t b = msb(keys[i], pass);
            local[b++];
        }

        #pragma omp for schedule(static)
        for (int b = 0; b < RADIX; b += 4) {
            __m256i sum = _mm256_setzero_si256(); // lets hope its a 64 bit platform

            for (int t = 0; t < nthreads; t++) {
                __m256i tmp = _mm256_loadu_si256((__m256i*)&thist[t * RADIX + b]);
                sum = _mm256_add_epi64(sum, tmp);
            }

            _mm256_storeu_si256((__m256i*)&hist[b], sum);
        }
    }

    free(thist);
}

void compute_heads_tails(Py_ssize_t* hist, Py_ssize_t* heads, Py_ssize_t* tails) {
    Py_ssize_t sum = 0;

    for (int b = 0; b < RADIX; b++) {
        heads[b] = sum;
        sum += hist[b];
        tails[b] = sum;
    }
}

// had to get some help from the big GPT for this
void permute(uint64_t* keys, Py_ssize_t* indices, int pass, Py_ssize_t* heads, Py_ssize_t* tails, Py_ssize_t* hist) {
    int P = omp_get_max_threads();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Py_ssize_t php[RADIX];
        Py_ssize_t ptp[RADIX];

        for (int b = 0; b < RADIX; b++) {
            Py_ssize_t base  = heads[b];
            Py_ssize_t size  = hist[b];
            Py_ssize_t chunk = size / P;
            Py_ssize_t rem   = size % P;

            php[b] = base + tid * chunk + (tid < rem ? tid : rem);
            ptp[b] = php[b] + chunk + (tid < rem);
        }

        for (int b = 0; b < RADIX; b++) {
            Py_ssize_t head = php[b];
            while (head < ptp[b]) {
                uint64_t v = keys[head];
                Py_ssize_t idx = indices[head];
                uint8_t k = msb(v, pass);

                while (k != b && php[k] < ptp[k]) {
                    Py_ssize_t dst = php[k]++;

                    uint64_t tmpk = keys[dst];
                    keys[dst] = v;
                    v = tmpk;

                    Py_ssize_t tmpi = indices[dst];
                    indices[dst] = idx;
                    idx = tmpi;

                    k = msb(v, pass);
                }

                keys[head] = v;
                indices[head] = idx;
                head++;
            }
        }
    }
}

void repair(uint64_t* keys, Py_ssize_t* indices, int pass, Py_ssize_t* heads, Py_ssize_t* tails) {
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < RADIX; b++) {
        Py_ssize_t l = heads[b];
        Py_ssize_t r = tails[b] - 1;

        while (l < r) {
            while (l < r && msb(keys[l], pass) == b) l++;
            while (l < r && msb(keys[r], pass) != b) r--;

            if (l < r) {
                uint64_t tk = keys[l];
                keys[l] = keys[r];
                keys[r] = tk;

                Py_ssize_t ti = indices[l];
                indices[l] = indices[r];
                indices[r] = ti;

                l++; r--;
            }
        }
    }
}



// stanford paper: https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/he.pdf
// more math heavy paper: https://scispace.com/pdf/paradis-an-efficient-parallel-algorithm-for-in-place-radix-4zo3nn8gi4.pdf
void radix_64_inp_par(uint64_t* keys, Py_ssize_t* indices, Py_ssize_t N, int pass) {
    if (N <= 1 || pass >= PASSES) {
        return;
    }

    Py_ssize_t hist[RADIX] = {0};
    Py_ssize_t heads[RADIX], tails[RADIX];

    build_histogram(keys, N, pass, hist);
    compute_heads_tails(hist, heads, tails);
    permute(keys, indices, pass, heads, tails, hist);
    repair(keys, indices, pass, heads, tails);

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < RADIX; b++) {
        Py_ssize_t start = heads[b];
        Py_ssize_t size = tails[b] - heads[b];
        if (size > 1) {
            radix_64_inp_par(keys + start, indices + start, size, pass + 1);
        }
    }    
}


// my naive impl
void radix_sort_64(uint64_t* keys, Py_ssize_t* indices,  Py_ssize_t N) {
    if (N <= 1) return;

    uint64_t* tmp_keys = malloc(N * sizeof(*tmp_keys));
    Py_ssize_t* tmp_indices = malloc(N * sizeof(*tmp_indices));
    if (!tmp_keys || !tmp_indices) {
        free(tmp_keys);
        free(tmp_indices);
        return;
    }

    Py_ssize_t count[RADIX];

    uint64_t* in_keys = keys;
    Py_ssize_t* in_idx = indices;
    uint64_t* out_keys = tmp_keys;
    Py_ssize_t* out_idx = tmp_indices;

    for (int p = 0; p < PASSES; p++) {
        memset(count, 0, sizeof(count));

        int shift = p * 8;
        for (Py_ssize_t i = 0; i < N; i++) {
            uint8_t byte = (in_keys[i] >> shift) & 0xFF;
            count[byte]++;
        }

        Py_ssize_t sum = 0;
        for (int i = 0; i < RADIX; i++) {
            Py_ssize_t tmp = count[i];
            count[i] = sum;
            sum += tmp;
        }

        for (Py_ssize_t i = 0; i < N; i++) {
            uint8_t byte = (in_keys[i] >> shift) & 0xFF;
            out_keys[count[byte]] = in_keys[i];
            out_idx[count[byte]] = in_idx[i];
            count[byte]++;
        }

        // swap in and out
        uint64_t* tmpk = in_keys; in_keys = out_keys; out_keys = tmpk;
        Py_ssize_t* tmpi = in_idx; in_idx = out_idx; out_idx = tmpi;
    }

    if (in_keys != keys) {
        memcpy(keys, in_keys, N * sizeof(*keys));
        memcpy(indices, in_idx, N * sizeof(*indices));
    }

    free(tmp_keys);
    free(tmp_indices);
}