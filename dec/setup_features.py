from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name = "features",
  ext_modules = cythonize("features.pyx"),
  include_dirs = [np.get_include()]
)
