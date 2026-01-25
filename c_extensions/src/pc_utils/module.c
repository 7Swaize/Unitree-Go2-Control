#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "methods.h"


PyObject* decode_xyz_intensity(PyObject*, PyObject*);
PyObject* apply_filter(PyObject*, PyObject*);

static PyMethodDef module_methods[] = {
    FAST_PC_DECODE_METHODS,
    FAST_PC_FILTER_METHODS,
    FAST_PC_METHODS_END
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fast_pointcloud",
    "Fast pointcloud decode + filtering",
    -1,
    module_methods
};


PyMODINIT_FUNC PyInit_fast_pointcloud(void) {
    import_array();
    return PyModule_Create(&moduledef);
}