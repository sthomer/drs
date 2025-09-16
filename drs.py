import numpy as np
import scipy.la as la
import eiscor

# def fpt(cs: np.ndarray) -> np.ndarray:


def fit_q_poly(cs: np.ndarray, K: int) -> np.ndarray:

    assert K <= len(cs) // 2
    try:
        q = la.solve_toeplitz((cs[K : 2 * K], np.flip(cs[1 : K + 1])), -cs[:K])

        return np.insert(np.flip(q), 0, 1)

    except la.LinAlgError as e:

        A = la.hankel(cs[1 : K + 1], cs[K : 2 * K])
        b = -cs[:K]
        qs_ = la.lstsq(A, b)[0]

        return np.insert(qs_, 0, 1)


def fit_p_poly(cs: np.ndarray, qs: np.ndarray, K: int) -> np.ndarray:

    assert K <= len(cs) // 2

    A = np.array([np.append([0] * i, cs[: K - i]) for i in range(K)])
    p = A @ qs[1:]

    return np.insert(p, 0, 0)


def poly_roots(qs: np.ndarray) -> np.ndarray:

    K = len(qs)
    qs_ = np.asfortranarray(np.flip(qs), dtype=np.complex128)
    zs_ = np.zeros(K - 1, dtype=np.complex128, order="F")
    rs_ = np.zeros(K - 1, dtype=np.float64, order="F")
    i_ = np.array(0)
    eiscor.z_poly_roots(qs_, zs_, rs_, i_)

    return np.ascontiguousarray(zs_)


def resonant_amplitudes(
    cs: np.ndarray, qs: np.ndarray, ps: np.ndarray, zs: np.ndarray, K: int
) -> np.ndarray:

    assert len(zs) <= len(cs) // 2 <= K

    i = np.arange(1, K + 1)
    exponent = np.tile(i, (K, 1))
    Z = np.power(zs.reshape(-1, 1), exponent)
    if len(ps) > K + 1 or len(qs) > K + 1:
        ps = ps[: K + 1]
        qs = qs[: K + 1]
    else:
        ps = ps
        qs = qs
    d = np.divide(Z @ ps[1:].reshape(-1, 1), Z @ (qs[1:] * i).reshape(-1, 1))

    return d.reshape((-1,))


def resonant_frequencies(zs: np.ndarray, sample_rate: int) -> np.ndarray:
    return sample_rate * 1j * np.log(zs)


def fpt(cs: np.ndarray, K: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:

    qs = fit_q_poly(cs, K)
    ps = fit_p_poly(cs, qs, K)
    zs = poly_roots(qs)
    ds = resonant_amplitudes(cs, qs, ps, zs, K)
    ws = resonant_frequencies(zs, sample_rate)

    return ds, ws, zs


def bifpt(cs: np.ndarray, K: int, sample_rate: int) -> np.ndarray:

    ds_in_, ws_in_, zs_in_ = fpt(cs, K, sample_rate)
    ds_in = ds_in_[ws_in_.imag < 0]
    ws_in = ws_in_[ws_in_.imag < 0]
    zs_in = zs_in_[ws_in_.imag < 0]

    ds_rev, ws_rev, zs_rev = fpt(np.flip(cs), K, sample_rate)
    ds_out_ = ds_rev * zs_rev ** (len(cs) - 1)
    ws_out_ = -ws_rev
    zs_out_ = 1 / zs_rev
    ds_out = ds_out_[ws_out_.imag > 0]
    ws_out = ws_out_[ws_out_.imag > 0]
    zs_out = zs_out_[ws_out_.imag > 0]

    ds = np.concatenate((ds_in, ds_out))
    ws = np.concatenate((ws_in, ws_out))
    zs = np.concatenate((zs_in, zs_out))

    return ds, ws, zs
