#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "methods.h"


PyObject* decode_xyz_intensity(PyObject*, PyObject*);
PyObject* apply_filter(PyObject*, PyObject*);

static PyObject* create_point_field_type_enum(void) {
    PyObject* enum_dict = PyDict_New();
    if (!enum_dict) return NULL;

    PyObject* int_int8 = PyLong_FromLong(1);
    PyObject* int_uint8 = PyLong_FromLong(2);
    PyObject* int_int16 = PyLong_FromLong(3);
    PyObject* int_uint16 = PyLong_FromLong(4);
    PyObject* int_int32 = PyLong_FromLong(5);
    PyObject* int_uint32 = PyLong_FromLong(6);
    PyObject* int_float32 = PyLong_FromLong(7);
    PyObject* int_float64 = PyLong_FromLong(8);

    if (!(int_int8 && int_uint8 && int_int16 && int_uint16 && int_int32 && int_uint32 && int_float32 && int_float64)) {
        Py_XDECREF(int_int8);
        Py_XDECREF(int_uint8);
        Py_XDECREF(int_int16);
        Py_XDECREF(int_uint16);
        Py_XDECREF(int_int32);
        Py_XDECREF(int_uint32);
        Py_XDECREF(int_float32);
        Py_XDECREF(int_float64);
        Py_DECREF(enum_dict);
        return NULL;
    }

    PyDict_SetItemString(enum_dict, "INT8", int_int8);
    PyDict_SetItemString(enum_dict, "UINT8", int_uint8);
    PyDict_SetItemString(enum_dict, "INT16", int_int16);
    PyDict_SetItemString(enum_dict, "UINT16", int_uint16);
    PyDict_SetItemString(enum_dict, "INT32", int_int32);
    PyDict_SetItemString(enum_dict, "UINT32", int_uint32);
    PyDict_SetItemString(enum_dict, "FLOAT32", int_float32);
    PyDict_SetItemString(enum_dict, "FLOAT64", int_float64);

    Py_DECREF(int_int8);
    Py_DECREF(int_uint8);
    Py_DECREF(int_int16);
    Py_DECREF(int_uint16);
    Py_DECREF(int_int32);
    Py_DECREF(int_uint32);
    Py_DECREF(int_float32);
    Py_DECREF(int_float64);

    return enum_dict;
}

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
    PyObject* module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    PyObject* point_field_enum = create_point_field_type_enum();
    if (!point_field_enum) {
        Py_DECREF(module);
        return NULL;
    }

    // Docs outline appropriate ref counting: https://docs.python.org/3/c-api/module.html
    if (PyModule_AddObjectRef(module, "PointFieldType", point_field_enum) < 0) {
        Py_DECREF(point_field_enum);
        Py_DECREF(module);
        return NULL;
    }

    Py_DECREF(point_field_enum);
    return module;
}