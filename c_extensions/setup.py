import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# https://stackoverflow.com/questions/27817190/what-does-cmdclass-do-in-pythons-setuptools
class build_ext_with_numpy(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)

        import numpy
        self.include_dirs.append(numpy.get_include())


extra_compile_args = ["-fopenmp"]
extra_link_args = ["-fopenmp"]

if os.environ.get("DEBUG") == "1":
    extra_compile_args += [
        "-O0",
        "-g",
        "-fno-omit-frame-pointer",
        "-fsanitize=address,undefined",
    ]
    extra_link_args += [
        "-fsanitize=address,undefined",
    ]
else:
    extra_compile_args += ["-O3"]


# https://numpy.org/devdocs/user/c-info.ufunc-tutorial.html
ext_modules = [
    Extension(
        "fast_pointcloud",
        sources=[
            os.path.join("src", "pc_utils", "module.c"),
            os.path.join("src", "pc_utils", "pcdecode.c"),
            os.path.join("src", "pc_utils", "pcfilter.c"),
            os.path.join("src", "utils", "atomic_bitset.c"),
            os.path.join("src", "utils", "sorting.c"),
        ],
        include_dirs=[
            os.path.join("src", "pc_utils"),
            os.path.join("src", "utils"),
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_2_0_API_VERSION')],
        extra_compile_args=extra_compile_args, 
        extra_link_args=extra_link_args
    )
]


setup(
    cmdclass={'build_ext': build_ext_with_numpy},
    ext_modules=ext_modules,
)