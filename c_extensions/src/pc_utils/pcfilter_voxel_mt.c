#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#include <atomic_bitset.h>

typedef struct {
    double max_range;
    double min_range;
    double height_min;
    double height_max;
    int downsample_rate;
    double sor_radius;
    int sor_min_neighbors;
    double intensity_min;
} FilterConfig;


static inline uint64_t voxel_hash(int64_t x, int64_t y, int64_t z) {
    uint64_t hash = (uint64_t)(x * 73856093LL) ^ (uint64_t)(y * 19349663LL) ^ (uint64_t)(z * 83492791LL);
    return hash + 1;
}


static inline bool get_attr_float(PyObject* obj, const char* name, double* out) {
    PyObject* tmp = PyObject_GetAttrString(obj, name);
    if (!tmp) return false;
    *out = PyFloat_AsDouble(tmp);
    Py_DECREF(tmp);
    if (PyErr_Occurred()) return false;
    return true;
}

static inline bool get_attr_int(PyObject* obj, const char* name, int* out) {
    PyObject* tmp = PyObject_GetAttrString(obj, name);
    if (!tmp) return false;
    *out = PyLong_AsLong(tmp);
    Py_DECREF(tmp);
    if (PyErr_Occurred()) return false;
    return true;
}

// returns 'true' if success, 'false' if failure
#define GET_ATTR(obj, field, out) \
    _Generic((out), \
        double*: get_attr_float, \
        int*:    get_attr_int \
    )(obj, field, out)


// some paper on parallelization: https://www.sci.utah.edu/~csilva/papers/cgf.pdf
void radix_sort_64(uint64_t* keys, Py_ssize_t* indices,  Py_ssize_t N) {
    if (N <= 1) return;

    uint64_t* tmp_keys = malloc(N * sizeof(*tmp_keys));
    Py_ssize_t* tmp_indices = malloc(N * sizeof(*tmp_indices));
    if (!tmp_keys || !tmp_indices) {
        free(tmp_keys);
        free(tmp_indices);
        return;
    }

    const int RADIX = 256; 
    const int PASSES = 8;
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

void radix_sort_64_mt(uint64_t* keys, Py_ssize_t* indices, Py_ssize_t N) {
    if (N <= 1) return;

    uint64_t* tmp_keys = malloc(N * sizeof(*tmp_keys));
    Py_ssize_t* tmp_indices = malloc(N * sizeof(*tmp_indices));
    if (!tmp_keys || !tmp_indices) {
        free(tmp_keys);
        free(tmp_indices);
        return;
    }

    const int RADIX = 256;
    const int PASSES = 8;

    uint64_t* in_keys = keys;
    Py_ssize_t* in_idx = indices;
    uint64_t* out_keys = keys;
    Py_ssize_t* out_idx = indices;

    int nthreads = omp_get_max_threads();
    Py_ssize_t* thread_count = malloc(nthreads * RADIX * sizeof(*thread_count));
    if (!thread_count) {
        free(tmp_keys);
        free(tmp_indices);
        return;
    }

    for (int p = 0; p < PASSES; p++) {
        int shift = p * 8;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            Py_ssize_t* hist = thread_count + (tid * RADIX);
            memset(hist, 0, RADIX * sizeof(Py_ssize_t));

            #pragma omp for schedule(static)
            for (Py_ssize_t i = 0; i < N; i++) {
                uint8_t byte = (in_keys[i] >> shift) & 0xFF;
                hist[byte]++;
            } 
        }

        Py_ssize_t global_count[RADIX] = {0};

        for (int b = 0; b < RADIX; ++b) {
            for (int t = 0; t < nthreads; ++t) {
                Py_ssize_t tmp = thread_count[t*RADIX + b];
                thread_count[t*RADIX + b] = global_count[b];
                global_count[b] += tmp;
            }
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            Py_ssize_t* hist = thread_count + tid * RADIX;

            #pragma omp for schedule(static)
            for (Py_ssize_t i = 0; i < N; ++i) {
                uint8_t byte = (in_keys[i] >> shift) & 0xFF;
                Py_ssize_t pos;

                #pragma omp atomic capture
                pos = global_count[byte]++;
                
                out_keys[pos] = in_keys[i];
                out_idx[pos]  = in_idx[i];
            }
        }

        uint64_t* tmpk = in_keys; in_keys = out_keys; out_keys = tmpk;
        Py_ssize_t* tmpi = in_idx; in_idx = out_idx; out_idx = tmpi;
    }

    if (in_keys != keys) {
        memcpy(keys, in_keys, N * sizeof(*keys));
        memcpy(indices, in_idx, N * sizeof(*indices));
    }

    free(tmp_keys);
    free(tmp_indices);
    free(thread_count);
}



static PyObject* apply_filter(PyObject* self, PyObject* args) {
    PyArrayObject* points_obj;
    PyObject* config_obj;

    if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &points_obj, &config_obj)) {
        return NULL;
    }

    // https://docs.scipy.org/doc/numpy-1.11.0/reference/c-api.array.html
    if (PyArray_TYPE(points_obj) != NPY_DOUBLE || PyArray_NDIM(points_obj) != 2) {
        return PyErr_Format(PyExc_TypeError, "Input point array must be 2D of dtype float64");
    }

    Py_ssize_t N = PyArray_DIM(points_obj, 0);
    Py_ssize_t D = PyArray_DIM(points_obj, 1);
    if (D < 3) return PyErr_Format(PyExc_ValueError, "Input point array must have atleast 3 columns (xyz)");

    double* points_buf = (double*)PyArray_DATA(points_obj); // borrowed ref

    FilterConfig cfg;
    if (!GET_ATTR(config_obj, "max_range", &cfg.max_range)) return NULL;
    if (!GET_ATTR(config_obj, "min_range", &cfg.min_range)) return NULL;
    if (!GET_ATTR(config_obj, "height_filter_min", &cfg.height_min)) return NULL;
    if (!GET_ATTR(config_obj, "height_filter_max", &cfg.height_max)) return NULL;
    if (!GET_ATTR(config_obj, "downsample_rate", &cfg.downsample_rate)) return NULL;
    if (!GET_ATTR(config_obj, "sor_radius", &cfg.sor_radius)) return NULL;
    if (!GET_ATTR(config_obj, "sor_min_neighbors", &cfg.sor_min_neighbors)) return NULL;
    if (!GET_ATTR(config_obj, "intensity_min", &cfg.intensity_min)) return NULL;

    double rmin2 = cfg.min_range * cfg.min_range;
    double rmax2 = cfg.max_range * cfg.max_range;
    double voxel_size = cfg.sor_radius;

    AtomicBitset* bs = bitset_create(N); // need to free
    bitset_clear_all(bs);
    Py_ssize_t keep_count = 0;

    Py_BEGIN_ALLOW_THREADS
    // basically splits threads over a (fairly) equal workload and handles atomic update of 'keep_count'
    #pragma omp for schedule(static) reduction(+: keep_count)
    for (Py_ssize_t i = 0; i < N; i += cfg.downsample_rate) {
        double x = points_buf[i*D + 0];
        double y = points_buf[i*D + 1];
        double z = points_buf[i*D + 2];
        double r2 = x*x + y*y + z*z;

        if (r2 < rmin2 || r2 > rmin2) continue;
        if (z < cfg.height_min || z > cfg.height_max) continue;

        if (D > 3) {
            double inten = points_buf[i*D + 3];
            if (inten < cfg.intensity_min) continue;
        }

        bitset_set_relaxed(bs, i);
        keep_count++;
    }

    // later: https://stackoverflow.com/questions/43057426/openmp-multiple-threads-update-same-array
    // also multi thread later

    Py_ssize_t* indices = malloc(keep_count * sizeof(*indices)); // need to free
    Py_ssize_t pos = 0;

    for (Py_ssize_t i = 0; i < N; i++) {
        if (bitset_test(bs, i)) {
            indices[pos++] = i;
        }
    }


    uint64_t* voxel_keys = malloc(keep_count * sizeof(*voxel_keys)); // need to free
    // shouldnt need any lock or safety because unique indices are garunteed
    #pragma omp parallel for schedule(static)
    for (Py_ssize_t i = 0; i < keep_count; i++) {
        Py_ssize_t idx = indices[i];

        double x = points_buf[idx*D + 0];
        double y = points_buf[idx*D + 1];
        double z = points_buf[idx*D + 2];

        int64_t ix = (int64_t)floor(x / cfg.sor_radius);
        int64_t iy = (int64_t)floor(y / cfg.sor_radius);
        int64_t iz = (int64_t)floor(z / cfg.sor_radius);

        voxel_keys[i] = voxel_hash(ix, iy, iz);
    }

    radix_sort_64(voxel_keys, indices, keep_count);
}
