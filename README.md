# Discrete Resonance Spectrogram (DRS)

## Installation

- You only have to do this once!
- Run `make install` to compile and link the fortran source.
    - Requirements
        - `gfortran` compiler to compile the fortran. (Install with `brew install gcc`)
        - `numpy.f2py` module to wrap and link fortain. (Install with `pip install numpy`)
- The generated library files (`.dylib`, `.so`) must be in the pythonpath.
    - Easiest way is to keep them in the same directory as the python file.

## Usage

- Call `bifpt(cs, K)` to calculate the resonance decomposition of the signal `cs`.
    - If not provided, `K` defaults to `len(cs) // 2`
    - The results is a matrix of shape `(2, K)`.
        - The first row contains the resonant amplitudes ($d_k$).
        - The second row contains the corresponding poles ($z_k$).
        - To separate them, call `ds, zs = np.unstack(bifpt(cs, K))`.
    - You can convert the poles to resonant frequencies with `resonant_frequency(z, sample_rate)`
- Call `reconstruction(ds, zs, N)` to calculate the signal from the given resonances.
