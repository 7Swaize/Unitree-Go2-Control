#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#include "atomic_bitset.h"
#include "sorting.h"

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

    

static Py_ssize_t calc_post_init_fcnt(double* points_buf, AtomicBitset* bs, FilterConfig* cfg, Py_ssize_t N, Py_ssize_t D) {
    double rmin2 = cfg->min_range * cfg->min_range;
    double rmax2 = cfg->max_range * cfg->max_range;

    Py_ssize_t post_init_fcnt = 0;

    #pragma omp parallel for schedule(static) reduction(+:post_init_fcnt)
    for (Py_ssize_t i = 0; i < N; i += cfg->downsample_rate) {
        double x = points_buf[i*D + 0];
        double y = points_buf[i*D + 1];
        double z = points_buf[i*D + 2];
        double r2 = x*x + y*y + z*z;

        if (r2 < rmin2 || r2 > rmax2) continue;
        if (z < cfg->height_min || z > cfg->height_max) continue;

        if (D > 3) {
            double inten = points_buf[i*D + 3];
            if (inten < cfg->intensity_min) continue;
        }

        bitset_set_relaxed(bs, i);
        post_init_fcnt++;
    }

    return post_init_fcnt;
}

static void calc_voxel_keys(double* points_buf, uint64_t* voxel_keys, Py_ssize_t* indices, FilterConfig* cfg, Py_ssize_t post_init_fcnt, Py_ssize_t D) {
    #pragma omp parallel for schedule(static)
    for (Py_ssize_t i = 0; i < post_init_fcnt; i++) {
        Py_ssize_t idx = indices[i];

        double x = points_buf[idx*D + 0];
        double y = points_buf[idx*D + 1];
        double z = points_buf[idx*D + 2];

        int64_t ix = (int64_t)floor(x / cfg->sor_radius);
        int64_t iy = (int64_t)floor(y / cfg->sor_radius);
        int64_t iz = (int64_t)floor(z / cfg->sor_radius);

        voxel_keys[i] = voxel_hash(ix, iy, iz);
    }
}

static void calc_valid_idx(Py_ssize_t* indices, AtomicBitset* bs, Py_ssize_t N) {
    Py_ssize_t pos = 0;

    for (Py_ssize_t i = 0; i < N; i++) {
        if (bitset_test(bs, i)) {
            indices[pos++] = i;
        }
    }
}


static Py_ssize_t calc_out(double** out_ptr, double* points_buf, uint64_t* voxel_keys, Py_ssize_t* indices, AtomicBitset* bs, FilterConfig* cfg, Py_ssize_t post_init_fcnt, Py_ssize_t D) {
    Py_ssize_t post_sor_fcnt = 0;

    bitset_clear_all(bs);

    for (Py_ssize_t voxel_start = 0; voxel_start < post_init_fcnt;) {
        Py_ssize_t voxel_end = voxel_start + 1;
        while (voxel_end < post_init_fcnt && voxel_keys[voxel_end] == voxel_keys[voxel_start]) {
            voxel_end++;
        }

        Py_ssize_t size = voxel_end - voxel_start;
        if (size >= cfg->sor_min_neighbors) {
            for (Py_ssize_t i = voxel_start; i < voxel_end; i++) {
                bitset_set_relaxed(bs, i);
                post_sor_fcnt++;
            }
        }

        voxel_start = voxel_end;
    }

    double* out = malloc(post_sor_fcnt * D * sizeof(*out));
    if (!out) {
        *out_ptr = NULL;
        return -1;
    }

    Py_ssize_t pos = 0;
    for (Py_ssize_t i = 0; i < post_init_fcnt; i++) {
        if (bitset_test(bs, i)) {
            Py_ssize_t idx = indices[i];
            for (Py_ssize_t d = 0; d < D; d++) {
                out[D*pos + d] = points_buf[idx*D + d];
            }
            pos++;
        }
    }

    *out_ptr = out;
    return post_sor_fcnt;
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


    AtomicBitset* bs = bitset_create(N); // need to free
    if (!bs) return PyErr_NoMemory();
    bitset_clear_all(bs);

    Py_ssize_t post_init_fcnt;
    Py_BEGIN_ALLOW_THREADS
    post_init_fcnt = calc_post_init_fcnt(points_buf, bs, &cfg, N, D);
    Py_END_ALLOW_THREADS


    Py_ssize_t* indices = malloc(post_init_fcnt * sizeof(*indices)); // need to free
    if (!indices) { bitset_free(bs); return PyErr_NoMemory(); }

    calc_valid_idx(indices, bs, N);


    free(bs);
    bs = NULL;


    uint64_t* voxel_keys = malloc(post_init_fcnt * sizeof(*voxel_keys)); // need to free
    if (!voxel_keys) { free(indices); return PyErr_NoMemory(); }

    Py_BEGIN_ALLOW_THREADS
    calc_voxel_keys(points_buf, voxel_keys, indices, &cfg, post_init_fcnt, D);
    Py_END_ALLOW_THREADS

    Py_BEGIN_ALLOW_THREADS
    radix_64_inp_par(voxel_keys, indices, post_init_fcnt, 0);
    Py_END_ALLOW_THREADS

    
    bs = bitset_create(post_init_fcnt); // need to free
    if (!bs) { free(indices); free(voxel_keys); return PyErr_NoMemory(); }

    
    double* out_buf;
    Py_ssize_t post_sor_fcnt;
    Py_BEGIN_ALLOW_THREADS
    post_sor_fcnt = calc_out(&out_buf, points_buf, voxel_keys, indices, bs, &cfg, post_init_fcnt, D);
    Py_END_ALLOW_THREADS


    free(indices);
    free(voxel_keys);
    bitset_free(bs);

    if (post_sor_fcnt < 0 || out_buf) {
        return PyErr_NoMemory();
    } 


}
