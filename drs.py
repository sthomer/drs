from time import perf_counter

import matplotlib as mpl
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
    return sample_rate * 1j * np.log(zs) / (2 * np.pi)


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


def power(dzs, window_size):
    d, z = np.unstack(dzs)
    d2, z2, n = np.abs(d) ** 2, np.abs(z) ** 2, window_size
    return (d2 / n) * ((z2**n - 1) / (z2 - 1)) / n


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


def chunk_every(cs, window_size, step_size=None):
    step_size = step_size or window_size
    return [
        (offset, cs[offset : offset + window_size])
        for offset in range(0, len(cs), step_size)
        if offset + window_size <= len(cs)
    ]


def drs(cs, window_size, step_size=None):
    assert window_size % 2 == 0

    return [
        (fpt(signal), fpt(np.flip(signal)), offset, window_size)
        for offset, signal in chunk_every(cs, window_size, step_size)
    ]


def spectral_params(dzs, dzs_rev, window_size, sample_rate):
    ds_in, zs_in = np.unstack(dzs)
    ds_out, zs_out = np.unstack(mirror(dzs_rev, window_size))

    ws_in = resonant_frequencies(zs_in, sample_rate)
    ws_out = resonant_frequencies(zs_out, sample_rate)

    ds = np.concatenate((ds_in, ds_out))
    ws = np.concatenate((ws_in, ws_out))

    return np.stack((amplitude(ds), phase(ds), frequency(ws), decay(ws)))


def plot_drs(spectrogram, sample_rate, ylim=None):

    powers = sorted(
        [
            pow
            for dzs, _, _, window_size in spectrogram
            for pow in power(dzs, window_size)
        ]
        + [
            pow
            for _, dzs_rev, _, window_size in spectrogram
            for pow in power(dzs_rev, window_size)
        ]
    )
    vmin = powers[-len(powers) // 4]
    vmax = powers[-1]
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = mpl.cm.binary

    for dzs, dzs_rev, offset, window_size in spectrogram:
        _, z = np.unstack(dzs)
        _, z_rev = np.unstack(dzs_rev)
        zs = np.concatenate((z, z_rev))
        frequencies = np.real(resonant_frequencies(zs, sample_rate))
        powers = np.concatenate((power(dzs, window_size), power(dzs_rev, window_size)))
        for frequency, pow in zip(frequencies, powers):
            plt.plot(
                [offset, offset + window_size],
                [frequency, frequency],
                color=cmap(norm(pow)),
            )
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def from_file(filename, dir="./data/input", filetype="wav"):
    file_in = f"{dir}/{filename}.{filetype}"
    signal, sample_rate = sf.read(file_in)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # turn stereo into mono
    return signal, sample_rate


def to_wav(signal, filename="temp", sample_rate=44100, dir="./data/output"):
    sf.write(f"{dir}/{filename}.wav", signal.real, sample_rate)


def run(window_size, step_size):
    signal, sample_rate = from_file("zero")
    start = 0
    length = (len(signal) // 2) * 2  # N must be a multiple of 2
    cs = signal[start : start + length]

    start = perf_counter()
    result = drs(cs, window_size, step_size)
    print(perf_counter() - start)

    # recon = []
    # for dzs, dzs_rev, offset, window_size in result:
    #     r = reconstruction(dzs, dzs_rev, window_size)
    #     recon += r.tolist()

    plot_drs(result, sample_rate, ylim=(0, 5000))
