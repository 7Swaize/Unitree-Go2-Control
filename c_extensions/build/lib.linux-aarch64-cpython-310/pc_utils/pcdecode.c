#define PY_SSIZE_T_CLEAN
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL FPC_ARRAY_API
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "methods.h"

typedef enum {
    PF_INT8    = 1,
    PF_UINT8   = 2,
    PF_INT16   = 3,
    PF_UINT16  = 4,
    PF_INT32   = 5,
    PF_UINT32  = 6,
    PF_FLOAT32 = 7,
    PF_FLOAT64 = 8
} PointFieldType;


static int host_little_endian(void) {
    uint16_t x = 1;
    return *((uint8_t*)&x);
}

static inline double read_point_field(const char* p, PointFieldType dtype, int swap) {
    switch (dtype) {
        case PF_INT8:   return (double)(*(int8_t*)p);
        case PF_UINT8:  return (double)(*(uint8_t*)p);
        case PF_INT16: {
            int16_t val = *(int16_t*)p;
            if (swap) val = (int16_t)__builtin_bswap16((uint16_t)val);
            return (double)val;
        }
        case PF_UINT16: {
            uint16_t val = *(uint16_t*)p;
            if (swap) val = __builtin_bswap16(val);
            return (double)val;
        }
        case PF_INT32: {
            int32_t val = *(int32_t*)p;
            if (swap) val = (int32_t)__builtin_bswap32((uint32_t)val);
            return (double)val;
        }
        case PF_UINT32: {
            uint32_t val = *(uint32_t*)p;
            if (swap) val = __builtin_bswap32(val);
            return (double)val;
        }
        case PF_FLOAT32: {
            union { uint32_t u; float f; } tmp;
            tmp.u = *(uint32_t*)p;
            if (swap) tmp.u = __builtin_bswap32(tmp.u);
            return (double)tmp.f;
        }
        case PF_FLOAT64: {
            union { uint64_t u; double f; } tmp;
            tmp.u = *(uint64_t*)p;
            if (swap) tmp.u = __builtin_bswap64(tmp.u);
            return tmp.f;
        }
        default:
            return 0.0;
    }
}

static PyObject* build_empty_ret(bool has_intensity) {
    PyObject* xyz = PyArray_SimpleNew(2, ((npy_intp[]){0, 3}), NPY_FLOAT64);
    if (!xyz) return NULL;

    PyObject* intensity = Py_None;
    if (has_intensity) {
        intensity = PyArray_SimpleNew(1, (npy_intp[]){0}, NPY_FLOAT64);
        if (!intensity) {
            Py_DECREF(xyz);
            return NULL;
        }
    } else {
        Py_INCREF(Py_None);
    }

    PyObject* ret = PyTuple_Pack(2, xyz, intensity);
    Py_DECREF(xyz);
    Py_DECREF(intensity);

    return ret;
}


// unused attr to prevent compiler from giving that dumb warning
__attribute__((unused)) PyObject* decode_xyz_intensity(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int point_step, ox, oy, oz, oi;
    int is_bigendian, dtype_xyz_val, dtype_intensity_val, skip_nans;

    if (!PyArg_ParseTuple(args, "Oiiiiiiiii", 
            &data_obj, &point_step, &ox, &oy, &oz, &oi,
            &is_bigendian, &dtype_xyz_val, &dtype_intensity_val, &skip_nans)) {
        return NULL;
    }

    PointFieldType dtype_xyz = (PointFieldType)dtype_xyz_val;
    PointFieldType dtype_intensity = (PointFieldType)dtype_intensity_val;
    
    // https://users.pja.edu.pl/~error501/python-html/c-api/buffer.html
    Py_buffer buf;
    if (PyObject_GetBuffer(data_obj, &buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }

    if (buf.len == 0 || point_step <= 0) {
        PyBuffer_Release(&buf);
        return build_empty_ret(oi >= 0);
    }

    Py_ssize_t n_points = buf.len / point_step;
    char* base = (char*)buf.buf;
    int swap = (host_little_endian() == is_bigendian);

    // why we do this: https://docs.python.org/3/extending/extending.html#back-to-the-example
    // about ref counts: // https://docs.python.org/3/extending/extending.html#reference-counts
    PyObject* xyz = PyArray_SimpleNew(2, ((npy_intp[]){n_points, 3}), NPY_FLOAT64);
    PyObject* intensity = (oi >= 0) ? PyArray_SimpleNew(1, ((npy_intp[]){n_points}), NPY_FLOAT64) : Py_None;
    if (intensity == Py_None) Py_INCREF(Py_None);

    
    double* xyz_data = (double*)PyArray_DATA((PyArrayObject*)xyz);
    double* i_data = (oi >= 0) ? (double*)PyArray_DATA((PyArrayObject*)intensity) : NULL;

    Py_ssize_t count = 0; // num valid points

    Py_BEGIN_ALLOW_THREADS
    char* p_base = base;
    for (Py_ssize_t i = 0; i < n_points; i++, p_base += point_step) {
        char* p = p_base;

        double x = read_point_field(p + ox, dtype_xyz, swap);
        double y = read_point_field(p + oy, dtype_xyz, swap);
        double z = read_point_field(p + oz, dtype_xyz, swap);
        double inten_val = (i_data) ? read_point_field(p + oi, dtype_intensity, swap) : 0.0;

        if (skip_nans && (isnan(x) || isnan(y) || isnan(z) || (i_data && oi >= 0 && isnan(inten_val)))) {
            continue;
        }

        xyz_data[count * 3 + 0] = x;
        xyz_data[count * 3 + 1] = y;
        xyz_data[count * 3 + 2] = z;

        if (i_data && oi >= 0) {
            i_data[count] = inten_val;
        }
        
        count++;
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);

    if (count != n_points) {
        PyArray_Dims newshape = {(npy_intp[]){count, 3}, 2};
        PyArray_Resize((PyArrayObject*)xyz, &newshape, 1, NPY_CORDER);

        if (i_data) {
            PyArray_Dims newshape_i = {(npy_intp[]){count}, 1};
            PyArray_Resize((PyArrayObject*)intensity, &newshape_i, 1, NPY_CORDER);
        }
    }

    PyObject* ret = PyTuple_Pack(2, xyz, intensity);
    Py_DECREF(xyz); // i hope this ref tracking is correct
    Py_DECREF(intensity);

    return ret;
}
