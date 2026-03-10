try:
    from . import _eiscor_ext
except ImportError as e:
    raise ImportError(
        "The compiled extension '_eiscor_ext' could not be imported. "
        "Reinstall the package with: python -m pip install --no-build-isolation -e ."
    ) from e


import numpy as np
import scipy.linalg as la

def hankel(s, K):
    return np.block([s[k : k + K] for k in range(K)])


def fit_q_matpoly(s, K):
    A = hankel(s, K)
    b = s[K : 2 * K]
    Qs = la.lstsq(A, b)
    if Qs is not None:
        return Qs[0].T
    else:
        raise la.LinAlgError


def companion(Q, K, D):
    return np.block([[np.zeros((D * (K - 1), D)), np.eye(D * (K - 1))], [Q]])


def polyeig(Q, K, D):
    wv = la.eig(companion(Q, K, D), right=True)
    return wv[0], wv[1][:D]


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
    _eiscor_ext.z_poly_roots(qs_, zs_, rs_, i_)
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
