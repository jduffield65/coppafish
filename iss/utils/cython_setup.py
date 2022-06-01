from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Run in terminal each time change cython_morphology.pyx:
# cd ISS/utils/
# python cython_setup.py build_ext --inplace
# cython cython_morphology.pyx -a produces html document indicating python interaction.

setup(ext_modules=cythonize('cython_morphology.pyx', language_level="3"), package_dir={'iss': '..'},
      include_dirs=[np.get_include()])
