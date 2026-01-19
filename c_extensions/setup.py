import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# https://stackoverflow.com/questions/27817190/what-does-cmdclass-do-in-pythons-setuptools
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)

        import numpy
        self.include_dirs.append(numpy.get_include())


# https://numpy.org/devdocs/user/c-info.ufunc-tutorial.html
ext_modules = [
    Extension(
        "fast_pointcloud",
        sources=[
            os.path.join("src", "pc_utils", "pcdecode.c")
        ]
    )
]


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)