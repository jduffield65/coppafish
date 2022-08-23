# coppafish



## Installation
*coppafish* supports python 3.8 and above. It can be installed using pip:

``` bash
pip install coppafish
```

To use the napari [Viewer](view_results.md) and matplotlib [diagnostics](view_results.md#diagnostics), 
the *plotting* version must be installed:

``` bash
pip install coppafish[plotting]
```

To use the [optimised](code/omp/coefs.md#optimised) code, which is recommended for running the 
[*find spots*](pipeline/find_spots.md) and [*OMP*](pipeline/omp.md) sections of the pipeline (otherwise
they can be very slow), the *optimised* version must be installed:

``` bash
pip install coppafish[optimised]
```

!!! warning "Installing on Windows"

    The optimised code requires *jax* which is not supported on Windows, thus the *optimised* version 
    of *coppafish* cannot be used on Windows.

The *optimised* and *plotting* features can both be installed by running:

``` bash
pip install coppafish[optimised,plotting]
```
