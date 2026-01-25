#ifndef MERTHODS_H
#define METHODS_H

#include <Python.h>

// https://docs.python.org/3/extending/extending.html#the-module-s-method-table-and-initialization-function
#define FAST_PC_DECODE_METHODS \
    { \
        "decode_xyz_intensity", \
        decode_xyz_intensity, \
        METH_VARARGS, \
        "Fast PointCloud2 XYZ(+intensity) decoder" \
    }

#define FAST_PC_FILTER_METHODS \
    { \
        "apply_filter", \
        apply_filter, \
        METH_VARARGS, \
        "Apply full height, range, intensity, and SOR filtering" \
    }

#define FAST_PC_METHODS_END \
    { NULL, NULL, 0, NULL }


#endif