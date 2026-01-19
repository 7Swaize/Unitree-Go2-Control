#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>

#define PF_INT8    1
#define PF_UINT8   2
#define PF_INT16   3
#define PF_UINT16  4
#define PF_INT32   5
#define PF_UINT32  6
#define PF_FLOAT32 7
#define PF_FLOAT64 8


static int host_little_endian(void) {
    uint16_t x = 1;
    return *((uint8_t*)&x);
}

static inline double read_point_field(const char* p, int dtype, int swap) {
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


static PyObject* decode_xyz_intensity(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int point_step, ox, oy, oz, oi;
    int is_bigendian, dtype_xyz, dtype_intensity, skip_nans;

    if (!PyArg_ParseTuple(args, "Oiiiiiiiii", 
            &data_obj, &point_step, &ox, &oy, &oz, &oi,
            &is_bigendian, &dtype_xyz, &dtype_intensity, &skip_nans)) {
        return NULL;
    }
    
    // https://users.pja.edu.pl/~error501/python-html/c-api/buffer.html
    Py_buffer buf;
    if (PyObject_GetBuffer(data_obj, &buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }

    Py_ssize_t n_points = buf.len / point_step;
    char* base = (char*)buf.buf;

    int swap = (host_little_endian() == is_bigendian);

    npy_intp dims_xyz[2] = {n_points, 3};
    PyObject* xyz = PyArray_SimpleNew(2, dims_xyz, NPY_FLOAT64); 

    PyObject* intensity = Py_None;
    if (oi >= 0) {
        npy_intp dims_i[1] = {n_points};
        intensity = PyArray_SimpleNew(1, dims_i, NPY_FLOAT64);
    } else {
        // why we do this: https://docs.python.org/3/extending/extending.html#back-to-the-example
        // about ref counts: // https://docs.python.org/3/extending/extending.html#reference-counts
        Py_INCREF(Py_None);
    }

    
    double* xyz_data = (double*)PyArray_DATA((PyArrayObject*)xyz);
    double* i_data = oi >= 0 ? (double*)PyArray_DATA((PyArrayObject*)intensity) : NULL;

    Py_ssize_t count = 0; // num valid points

    Py_BEGIN_ALLOW_THREADS
    char* p_base = base;
    for (Py_ssize_t i = 0; i < n_points; i++, p_base += point_step) {
        char* p = p_base;

        double x = read_point_field(p + ox, dtype_xyz, swap);
        double y = read_point_field(p + oy, dtype_xyz, swap);
        double z = read_point_field(p + oz, dtype_xyz, swap);
        double inten_val = 0.0;

        if (i_data && oi >= 0) {
            inten_val = read_point_field(p + oi, dtype_intensity, swap);
        }

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
        PyArray_Dims newshape = {{count, 3}, 2};
        PyArray_Resize((PyArrayObject*)xyz, &newshape, 1, NPY_CORDER);

        if (i_data) {
            PyArray_Dims newshape_i = {{count}, 1};
            PyArray_Resize((PyArrayObject*)intensity, &newshape_i, 1, NPY_CORDER);
        }
    }

    PyObject* ret = PyTuple_Pack(2, xyz, intensity);
    Py_DECREF(xyz); // i hope this ref tracking is correct
    Py_DECREF(intensity);

    return ret ? ret : NULL;
}

// https://docs.python.org/3/extending/extending.html#the-module-s-method-table-and-initialization-function
static PyMethodDef methods[] = {
    {
        "decode_xyz_intensity",
        decode_xyz_intensity,
        METH_VARARGS,
        "Fast PointCloud2 XYZ(+intensity) decoder"
    },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fast_pointcloud",
    (PyCFunction)decode_xyz_intensity,
    -1,
    methods
};


PyMODINIT_FUNC PyInit_fast_pointcloud(void) {
    import_array() // i think for custom ret val on exception its "import_array1(ret)" but NULL is fine for us
    return PyModuleDef_Init(&moduledef);
}