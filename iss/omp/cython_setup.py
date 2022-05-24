from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Run in terminal each time change cython_morphology.pyx:
# cd ISS/omp/
# python cython_setup.py build_ext --inplace

setup(ext_modules=cythonize('cython_omp.pyx', language_level="3"), package_dir={'iss': '..'},
      include_dirs=[np.get_include()])
