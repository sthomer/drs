from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import soundfile as sf

import eiscor

NDArrayF64 = np.ndarray[tuple[int], np.dtype[np.float64]]
NDArrayC128 = np.ndarray[tuple[int], np.dtype[np.complex128]]


def fit_q_poly(cs: NDArrayF64, K: int) -> NDArrayF64:
    assert K <= cs.size // 2
    try:
        q = la.solve_toeplitz((cs[K : 2 * K], np.flip(cs[1 : K + 1])), -cs[:K])
        return np.real(np.insert(np.flip(q), 0, 1))
    except la.LinAlgError:
        A = la.hankel(cs[1 : K + 1], cs[K : 2 * K])
        b = -cs[:K]
        qs_ = la.lstsq(A, b)
        if qs_ is not None:
            return np.insert(qs_[0], 0, 1)
        else:
            raise la.LinAlgError


def fit_p_poly(cs: NDArrayF64, qs: NDArrayF64, K: int) -> NDArrayF64:
    assert K <= cs.size // 2
    A = np.array([np.append([0] * i, cs[: K - i]) for i in range(K)])
    p = A @ qs[1:]
    return np.insert(p, 0, 0)


def poly_roots(qs: NDArrayF64) -> NDArrayC128:
    qs_ = np.asfortranarray(np.flip(qs), dtype=np.complex128)
    zs_ = np.zeros(qs.size - 1, dtype=np.complex128, order="F")
    rs_ = np.zeros(qs.size - 1, dtype=np.float64, order="F")
    i_ = np.array(0)
    eiscor.z_poly_roots(qs_, zs_, rs_, i_)
    return np.ascontiguousarray(zs_)


def resonant_amplitudes(
    cs: NDArrayF64, qs: NDArrayF64, ps: NDArrayF64, zs: NDArrayC128, K: int
) -> NDArrayC128:
    assert zs.size <= K <= cs.size // 2
    Z = np.vander(zs, K + 1, increasing=True)[:, 1:]
    # if len(ps) > K + 1 or len(qs) > K + 1:
    #     ps = ps[: K + 1]
    #     qs = qs[: K + 1]
    numerator = Z @ ps[1:]
    denominator = Z @ (qs[1:] * np.arange(1, K + 1))
    d = numerator / denominator
    return d


def resonant_frequency(
    z: complex | NDArrayC128, sample_rate: int
) -> complex | NDArrayC128:
    return sample_rate * 1j * np.log(z)


def fpt(
    cs: NDArrayF64, K: int | None = None, atol: float = 1e-10
) -> tuple[NDArrayC128, NDArrayC128]:
    k = K or len(cs) // 2
    qs = fit_q_poly(cs, k)
    ps = fit_p_poly(cs, qs, k)
    zs_all = poly_roots(qs)
    zs_stable = zs_all[np.abs(zs_all) < 1]
    ds_stable = resonant_amplitudes(cs, qs, ps, zs_stable, k)
    ds_genuine = ds_stable[np.abs(ds_stable) > atol]
    zs_genuine = zs_stable[np.abs(ds_stable) > atol]
    return ds_genuine, zs_genuine


def amplitude(d: complex | NDArrayC128) -> float | NDArrayF64:
    return np.abs(d)


def phase(d: complex | NDArrayC128) -> float | NDArrayF64:
    return float(np.angle(d))


def frequency(z: complex | NDArrayC128, sample_rate: int) -> float | NDArrayF64:
    return np.real(resonant_frequency(z, sample_rate))


def decay(z: complex | NDArrayC128, sample_rate: int) -> float | NDArrayF64:
    return np.imag(resonant_frequency(z, sample_rate))


def mirror(
    d: NDArrayC128, z: NDArrayC128, N: int
) -> tuple[complex, complex] | tuple[NDArrayC128, NDArrayC128]:
    d_mirror = d * z ** (N - 1)
    z_mirror = 1 / z
    return d_mirror, z_mirror


def reconstruction(
    ds: NDArrayC128, zs: NDArrayC128, ds_rev: NDArrayC128, zs_rev: NDArrayC128, N: int
) -> NDArrayF64:
    V = np.vander(zs, N, increasing=True).T
    cs = np.real(V @ ds)
    V_rev = np.vander(zs_rev, N, increasing=True).T
    cs_rev = np.real(V_rev @ ds_rev)
    return cs + np.flip(cs_rev)


def drs(
    cs: NDArrayF64, window_length: int
) -> list[tuple[NDArrayC128, NDArrayC128, NDArrayC128, NDArrayC128, int, int]]:
    assert cs.size % 2 == 0
    assert window_length % 2 == 0
    num_windows = cs.size // window_length
    result = []
    for n in range(num_windows):
        offset = n * window_length
        signal = cs[offset : offset + window_length]
        ds, zs = fpt(signal)
        ds_rev, zs_rev = fpt(np.flip(signal))
        result.append((ds, zs, ds_rev, zs_rev, offset, window_length))
    offset = num_windows * window_length
    remains = cs[offset:]
    ds, zs = fpt(remains)
    ds_rev, zs_rev = fpt(np.flip(remains))
    result.append((ds, zs, ds_rev, zs_rev, offset, window_length))
    return result


def spectral_params(ds, zs, ds_rev, zs_rev, window_length, sample_rate):
    amplitudes_damp = amplitude(ds)
    phases_damp = phase(ds)
    frequencies_damp = frequency(zs, sample_rate)
    decays_damp = decay(zs, sample_rate)
    ds_ramp, zs_ramp = mirror(ds_rev, zs_rev, window_length)
    amplitudes_ramp = amplitude(ds_ramp)
    phases_ramp = phase(ds_ramp)
    frequencies_ramp = frequency(zs_ramp, sample_rate)
    decays_ramp = decay(zs_ramp, sample_rate)
    amplitudes = np.concatenate((amplitudes_damp, amplitudes_ramp))
    phases = np.concatenate((phases_damp, phases_ramp))
    frequencies = np.concatenate((frequencies_damp, frequencies_ramp))
    decays = np.concatenate((decays_damp, decays_ramp))
    return amplitudes, phases, frequencies, decays


def from_file(filename, dir="./data/input", filetype="wav"):
    file_in = f"{dir}/{filename}.{filetype}"
    signal, sample_rate = sf.read(file_in)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # turn stereo into mono
    return signal, sample_rate


def to_wav(signal, filename="temp", sample_rate=44100, dir="./data/output"):
    sf.write(f"{dir}/{filename}.wav", signal.real, sample_rate)


signal, sample_rate = from_file("zero")
start = 0
length = (len(signal) // 2) * 2  # N must be a multiple of 2
cs = signal[start : start + length]

window_size = 1024
result = drs(cs, window_size)
recon = []
spectrogram = []
for ds, zs, ds_rev, zs_rev, offset, window_length in result:
    amplitudes, phases, frequencies, decays = spectral_params(
        ds, zs, ds_rev, zs_rev, window_length, sample_rate
    )
    spectrogram.append((amplitudes, phases, frequencies, decays, offset, window_length))
    r = reconstruction(ds, zs, ds_rev, zs_rev, window_size)
    recon += r.tolist()

plt.plot(cs)
plt.plot(recon)
plt.show()
