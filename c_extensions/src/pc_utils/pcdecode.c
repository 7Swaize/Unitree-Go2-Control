#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>



static int host_little_endian(void) {
    uint16_t x = 1;
    return *((uint8_t*)&x);
}

static inline float load_32(const char* p, int swap) {
    union { uint32_t u; float f; } tmp;
    tmp.u = *(uint32_t*)p;
    if (swap) {
        tmp.u = __builtin_bswap32(tmp.u);
    }

    return tmp.f;
}

static inline double load_64(const char* p, int swap) {
    union { uint64_t u; double f; } tmp;
    tmp.u = *(uint64_t*)p;
    if (swap) {
        tmp.u = __builtin_bswap64(tmp.u);
    }

    return tmp.f;
}


static PyObject* decode_xyz_intensity(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int point_step, ox, oy, oz, oi;
    int is_bigendian, dtype, skip_nans;

    if (!PyArg_ParseTuple(args, "Oiiiiiiii", &data_obj, &point_step, &ox, &oy, &oz, &oi, &is_bigendian, &dtype, &skip_nans)) {
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

        double x = (dtype == 32) ? load_32(p + ox, swap) : load_64(p + ox, swap);
        double y = (dtype == 32) ? load_32(p + oy, swap) : load_64(p + oy, swap);
        double z = (dtype == 32) ? load_32(p + oz, swap) : load_64(p + oz, swap);
        double inten_val = 0.0;

        if (i_data && oi >= 0) {
            inten_val = (dtype == 32) ? load_32(p + oi, swap) : load_64(p + oi, swap);
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
    NULL, // docs later?
    -1,
    methods
};


PyMODINIT_FUNC PyInit_fast_pointcloud(void) {
    import_array() // i think for custom ret val on exception its "import_array1(ret)" but NULL is fine for us
    return PyModuleDef_Init(&moduledef);
}