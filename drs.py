from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import soundfile as sf

import eiscor


def fit_q_poly(cs, K):
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


def fit_p_poly(cs, qs, K):
    A = np.array([np.append([0] * i, cs[: K - i]) for i in range(K)])
    p = A @ qs[1:]
    return np.insert(p, 0, 0)


def poly_roots(qs):
    qs_ = np.asfortranarray(np.flip(qs), dtype=np.complex128)
    zs_ = np.zeros(qs.size - 1, dtype=np.complex128, order="F")
    rs_ = np.zeros(qs.size - 1, dtype=np.float64, order="F")
    i_ = np.array(0)
    eiscor.z_poly_roots(qs_, zs_, rs_, i_)
    return np.ascontiguousarray(zs_)


def resonant_amplitudes(qs, ps, zs, K):
    Z = np.vander(zs, K + 1, increasing=True)[:, 1:]
    numerator = Z @ ps[1:]
    denominator = Z @ (qs[1:] * np.arange(1, K + 1))
    ds = numerator / denominator
    return ds


def resonant_frequencies(zs, sample_rate):
    return sample_rate * 1j * np.log(zs)


def fpt(cs, K=None, atol=1e-10):
    k = K or len(cs) // 2
    qs = fit_q_poly(cs, k)
    ps = fit_p_poly(cs, qs, k)
    zs_all = poly_roots(qs)
    zs_stable = zs_all[np.abs(zs_all) < 1]
    ds_stable = resonant_amplitudes(qs, ps, zs_stable, k)
    ds_genuine = ds_stable[np.abs(ds_stable) > atol]
    zs_genuine = zs_stable[np.abs(ds_stable) > atol]
    return np.stack((ds_genuine, zs_genuine))


def amplitude(ds):
    return np.abs(ds)


def phase(ds):
    return np.angle(ds)


def frequency(ws):
    return np.real(ws)


def decay(ws):
    return np.imag(ws)


def mirror(dzs, N):
    ds, zs = np.unstack(dzs)
    ds_mirror = ds * zs ** (N - 1)
    zs_mirror = 1 / zs
    return np.stack((ds_mirror, zs_mirror))


def reconstruction(dzs, dzs_rev, N):
    ds, zs = np.unstack(dzs)
    V = np.vander(zs, N, increasing=True).T
    cs = np.real(V @ ds)
    ds_rev, zs_rev = np.unstack(dzs_rev)
    V_rev = np.vander(zs_rev, N, increasing=True).T
    cs_rev = np.real(V_rev @ ds_rev)
    return cs + np.flip(cs_rev)


def drs(cs, window_length):
    assert cs.size % 2 == 0
    assert window_length % 2 == 0
    num_windows = cs.size // window_length
    result = []
    for n in range(num_windows):
        offset = n * window_length
        signal = cs[offset : offset + window_length]
        dzs = fpt(signal)
        dzs_rev = fpt(np.flip(signal))
        result.append((dzs, dzs_rev, offset, window_length))
    offset = num_windows * window_length
    remains = cs[offset:]
    dzs = fpt(remains)
    dzs_rev = fpt(np.flip(remains))
    result.append((dzs, dzs_rev, offset, window_length))
    return result


def spectral_params(dzs, dzs_rev, window_length, sample_rate):
    ds_in, zs_in = np.unstack(dzs)
    ds_out, zs_out = np.unstack(mirror(dzs_rev, window_length))

    ws_in = resonant_frequencies(zs_in, sample_rate)
    ws_out = resonant_frequencies(zs_out, sample_rate)

    ds = np.concatenate((ds_in, ds_out))
    ws = np.concatenate((ws_in, ws_out))

    return np.stack((amplitude(ds), phase(ds), frequency(ws), decay(ws)))


def plot_drs(spectrogram, sample_rate):
    amplitudes, phases, frequencies, decays, offsets, window_length = [
        [b for a in x for b in a] for x in zip(*spectrogram)
    ]
    _, ax = plt.subplots()
    ax.scatter(
        np.array(offsets) / sample_rate,
        frequencies,
        c=amplitudes,
        cmap="binary",
        marker="_",
    )
    plt.show()


def from_file(filename, dir="./data/input", filetype="wav"):
    file_in = f"{dir}/{filename}.{filetype}"
    signal, sample_rate = sf.read(file_in)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # turn stereo into mono
    return signal, sample_rate


def to_wav(signal, filename="temp", sample_rate=44100, dir="./data/output"):
    sf.write(f"{dir}/{filename}.wav", signal.real, sample_rate)


signal, sample_rate = from_file("loop")
start = 0
length = (len(signal) // 2) * 2  # N must be a multiple of 2
cs = signal[start : start + length]

window_size = 2048
start = perf_counter()
result = drs(cs, window_size)
print(perf_counter() - start)

recon = []
for dzs, dzs_rev, offset, window_length in result:
    r = reconstruction(dzs, dzs_rev, window_size)
    recon += r.tolist()

plt.plot(cs)
plt.plot(recon)
plt.show()
