
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

"""
build command
>>> cd core/xface/extension
>>> python setup.py build_ext --inplace
>>> ls *.so
"""


setup(
    name='fusion_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            name='fusion_cython',
            sources=["fusion/fusion_cython.pyx", "fusion/fusion.cpp"],
            language='c++',
            extra_compile_args=["-fopenmp", "-msse2", "-mavx2"],
            extra_link_args=["-fopenmp", "-msse2", "-mavx2"],
            # extra_compile_args=["-fopenmp", "-mavx512f"],
            # extra_link_args=["-fopenmp", "-mavx512f"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
