import numpy as np
import scipy.linalg as la
import eiscor
import matplotlib.pyplot as plt
import soundfile as sf
from time import perf_counter

NDArrayF64 = np.ndarray[tuple[int], np.dtype[np.float64]]
NDArrayC128 = np.ndarray[tuple[int], np.dtype[np.complex128]]


def fit_q_poly(cs: NDArrayF64, K: int) -> NDArrayF64:
    # assert K <= cs.size // 2
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
    # assert K <= cs.size // 2
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
    # assert zs.size <= K <= cs.size // 2
    Z = np.vander(zs, K+1, increasing=True)[:, 1:]
    # if len(ps) > K + 1 or len(qs) > K + 1:
    #     ps = ps[: K + 1]
    #     qs = qs[: K + 1]
    numerator = Z @ ps[1:]
    denominator = Z @ (qs[1:] * np.arange(1, K+1))
    d = numerator / denominator
    return d


def resonant_frequency(z: NDArrayC128, sample_rate: int) -> NDArrayC128:
    return sample_rate * 1j * np.log(z)


def fpt(cs: NDArrayF64, K: int | None = None, atol: float = 1e-10) -> NDArrayC128:
    k = K or len(cs) // 2
    qs = fit_q_poly(cs, k)
    ps = fit_p_poly(cs, qs, k)
    zs_all = poly_roots(qs)
    zs_stable = zs_all[np.abs(zs_all) < 1]
    ds_stable= resonant_amplitudes(cs, qs, ps, zs_stable, k)
    ds_genuine = ds_stable[np.abs(ds_stable) > atol]
    zs_genuine = zs_stable[np.abs(ds_stable) > atol]
    return np.stack((ds_genuine, zs_genuine))


def mirror(ds_zs: NDArrayC128, N: int) -> NDArrayC128:
    ds, zs = np.unstack(ds_zs)
    ds_mirror = ds * zs ** (N - 1)
    zs_mirror = 1 / zs
    return np.stack((ds_mirror, zs_mirror))


def reconstruction(dzs: NDArrayC128, dzs_rev: NDArrayC128, N: int) -> NDArrayF64:
    ds, zs = np.unstack(dzs)
    V = np.vander(zs, N, increasing=True).T
    cs = np.real(V @ ds)

    ds_rev, zs_rev = np.unstack(dzs_rev)
    V_rev = np.vander(zs_rev, N, increasing=True).T
    cs_rev = np.real(V_rev @ ds_rev)

    return cs + np.flip(cs_rev)


def drs(cs: NDArrayF64, window_length: int) -> list[NDArrayC128]:
    assert cs.size % 2 == 0
    assert window_length % 2 == 0
    num_windows = cs.size // window_length
    result = []
    for n in range(num_windows):
        offset = n * window_length
        signal = cs[offset:offset+window_length]
        ds_zs = fpt(signal)
        ds_zs_rev = fpt(np.flip(signal))
        result.append((ds_zs, ds_zs_rev))
    # remains = cs[num_windows * window_length:]
    # ds_zs_remains = fpt(remains)
    # ds_zs_rev_remains = fpt(np.flip(remains))
    return result


def from_file(filename, dir='./data/input', filetype='wav'):
    file_in = f'{dir}/{filename}.{filetype}'
    signal, sample_rate = sf.read(file_in)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # turn stereo into mono
    return signal, sample_rate


def to_wav(signal, filename='temp', sample_rate=44100, dir='./data/output'):
    sf.write(f'{dir}/{filename}.wav', signal.real, sample_rate)


# signal, sample_rate = from_file("zero")
# start = 0
# length = (len(signal) // 2) * 2  # N must be a multiple of 2
# cs = signal[start:start+length] 
#
# ds_zs = fpt(cs)
# ds_zs_rev = fpt(np.flip(cs))
# recon = reconstruction(ds_zs, ds_zs_rev, length)
#
# window_size = 1024
# result = drs(cs, window_size)
# recons = []
# for ds_zs, ds_zs_rev in result:
#     r = reconstruction(ds_zs, ds_zs_rev, window_size)
#     recon += r.tolist()
#
# plt.plot(cs)
# plt.plot(recon)
# plt.show()

