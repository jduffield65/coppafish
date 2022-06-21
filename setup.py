from setuptools import setup

with open("iss/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()


setup(
    name = 'iss',
    version = __version__,
    description = 'In Situ Sequencing software for Python',
    long_description = long_desc,
    long_description_content_type='text/markdown',
    author = 'Josh Duffield',
    author_email = 'm.shinn@ucl.ac.uk',
    maintainer = 'Josh Duffield',
    maintainer_email = 'm.shinn@ucl.ac.uk',
    license = 'MIT',
    python_requires='>=3.8',
    url='https://github.com/mwshinn/spatiotemporal',
    packages = ['iss', 'iss.setup', 'iss.utils', 'iss.extract', 'iss.stitch', 'iss.spot_colors', 'iss.plot', 'iss.pipeline', 'iss.pcr', 'iss.omp', 'iss.find_spots', 'iss.call_spots'],
    install_requires = ['jax==0.3.10', 'jaxlib==0.3.10', 'numpy_indexed', 'tdqm', 'scipy', 'sklearn', 'tifffile', 'opencv-python', 'scikit-image', 'nd2', 'matplotlib', 'h5py'],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
)

