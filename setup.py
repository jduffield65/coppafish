from setuptools import setup, find_packages

with open("coppafish/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()

packages = [folder for folder in find_packages() if folder[-5:] != '.test']  # Get rid of test packages

setup(
    name='coppafish',
    version=__version__,
    description='coppaFISH software for Python',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    author='Josh Duffield',
    author_email='jduffield65@gmail.com',
    maintainer='Reilly Tilbury',
    maintainer_email='reillytilbury@gmail.com',
    license='MIT',
    python_requires='>=3.8',
    url='https://jduffield65.github.io/coppafish/',
    packages=packages,
    install_requires=['numpy', 'numpy_indexed', 'tqdm', 'scipy', 'sklearn', 'opencv-python-headless',
                      'scikit-image', 'nd2', 'h5py', 'pandas', 'cloudpickle', 'dask', 'joblib', 'threadpoolctl',
                      'cachey', 'sphinx'],
    extras_require={'optimised': ['jax', 'jaxlib'], 'plotting': ['matplotlib', 'distinctipy', 'PyQt5',
                                                                 'magicgui', 'ipympl', 'napari', 'npe2', 'hsluv',
                                                                 'freetype-py']},
    package_data={'coppafish.setup': ['settings.default.ini', 'notebook_comments.json',
                                      'dye_camera_laser_raw_intensity.csv'],
                  'coppafish.plot.results_viewer': ['cell_color.csv', 'cellClassColors.json', 'gene_color.csv']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
)
