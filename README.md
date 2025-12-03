# Discrete Resonance Spectrogram (DRS)

## Installation

- You only have to do this once!
- Run `make install` to compile and link the fortran source.
    - Requirements
        - `python` duh.
        - `gfortran` compiler to compile the fortran. (Install with `brew install gcc`)
        - `meson` build system. (Install with `brew install meson`)
        - `ninja` build system. (Install with `brew install ninja`)
        - `numpy.f2py` module to wrap and link fortan. (Install with `pip install numpy`)
- The generated library files (mac: `.dylib`, windows: `.dll`, or linux: `.so`) must be in the pythonpath.
    - Easiest way is to keep them in the same directory as the python file.

## Usage

- Call `drs(cs, window_length)` to calculate the resonance decomposition of the signal `cs`.
    - This results in a list of pairs `(ds_zs, ds_zs_rev)`, one for each slice.
    - The first element in the pair corresponds to the forward pass.
        - It is a matrix of shape `(2, K')`, where `K'` is the number of _damping_ resonances.
        - The first row contains the resonant amplitudes ($d_k$).
        - The second row contains the corresponding poles ($z_k$).
    - The second element in the pair corresponds to the backward pass.
        - It is a matrix of shape `(2, K')`, where `K'` is the number of _ramping_ resonances.
        - The first row contains the resonant amplitudes ($d_k$).
        - The second row contains the corresponding poles ($z_k$).
    - Note: `ds_zs_rev` is not transformed to forward time!
        - They correspond to the resonances of the backward signal.
        - To transform to forward time, use `mirror(ds_zs, N)`.
    - To separate, call `ds, zs = np.unstack(ds_zs)`.
    - Convert the poles `zs` to resonant frequencies with `resonant_frequency(z, sample_rate)`
- Call `reconstruction(ds_zs, ds_zs_rev, N)` to calculate the signal from the given resonances.
