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


static inline int64_t voxel_hash(int64_t x, int64_t y, int64_t z) {
    return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
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

    AtomicBitset* bs = bitset_create(N);
    bitset_clear_all(bs);
    int keep_count = 0;

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

    Py_ssize_t* indices = (Py_ssize_t*)malloc(keep_count * sizeof(Py_ssize_t));
    Py_ssize_t pos = 0;

    for (Py_ssize_t i = 0; i < N; i++) {
        if (bitset_test(bs, i)) {
            indices[pos++] = i;
        }
    }


    int64_t* voxel_keys = (int64_t*)malloc(keep_count * sizeof(int64_t));
    // shouldnt need any lock or safety because unique indices are garunteed
    #pragma omp for schedule(static)
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
}