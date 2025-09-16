# Calling Fortran in Python

- Calling Fortran code in Python can be made easy by using `f2py`.
- `f2py` (or `f2py3`) is bundled with `numpy` and does all the heavy lifting of wrapping Fortran.
- These instructions are for python version 3.10.4 using `distutils` and `setuptools`.
- At least on Mac, you can solve the following error by commenting out the offending line.
```
  File "/usr/local/lib/python3.11/site-packages/numpy/distutils/mingw32ccompiler.py", line 27, in <module>
    from distutils.msvccompiler import get_build_version as get_build_msvc_version
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

## Compile Fortran Source into Python Module

```python
python -m numpy.f2py -c -L"(pwd)"/eiscor -leiscor eiscor/src/complex_double/z_poly_roots.f90 -m eiscor_wrapper
```

- To invoke f2py, use `python -m numpy.f2py` or the `f2py` command
- The `-c` argument specifies the Fortran source code file
- The `-L` argument specifies the directory that contains a shared library for compilation
- The `-l` argument specifies the name of the library for compilation.
    - This file will have a `.a`, `.so`, or `.dylib` extension, which must be omitted.
    - The filename will begin with `lib`, which much be omitted.
        - E.g., The library file `libeiscor.dylib` becomes `-leiscor`
- The `-m` argument specifies the name of the created python module.

## Importing the Generated Python Module

- This will create a `.so` library file beginning with the name provided to the `-m` option.
- To import this module, simply call `import eiscor_wrapper` in Python.
    - Both the generated wrapper file and the specified library file must be in the path.
    - The easiest way to ensure they are visible is to place them in the same directory as the python module that is doing the import.

## Calling Fortran code in Python

- When invoking Fortran code, you must allocate scalar, arrays, and matrices for input/output.
    - For in/out scalars, initialize to `0` with `x = np.array(0)`.
    - For in/out arrays and matrices, initialize with `order='F'` 
        - Or convert a numpy array with `np.asfortranarray(arr)`.
    - Fortran `COMPLEX(8)` corresponds to `dtype=np.complex128`
    - Fortran `REAL(8)` corresponds to `dtype=np.float64`
    - Fortran `INTEGER` corresponds to `dtype=np.int32`
- It is wise to convert back to a numpy array with `np.ascontiguousarray(fortran_array)`.
    - Assuming that the rest of the code uses standard numpy arrays.

