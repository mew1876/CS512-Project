from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Project',
	ext_modules=cythonize("Project.pyx"),
	include_dirs=[numpy.get_include()])