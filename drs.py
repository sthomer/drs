import numpy as np
import scipy.linalg as la
import eiscor
import matplotlib.pyplot as plt

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
    i = np.arange(1, K + 1)
    exponent = np.tile(i, (K, 1))
    Z = np.power(zs.reshape(-1, 1), exponent)
    if len(ps) > K + 1 or len(qs) > K + 1:
        ps = ps[: K + 1]
        qs = qs[: K + 1]
    d = np.divide(Z @ ps[1:].reshape(-1, 1), Z @ (qs[1:] * i).reshape(-1, 1))
    return d.reshape((-1,))


def resonant_frequency(z: NDArrayC128, sample_rate: int) -> NDArrayC128:
    return sample_rate * 1j * np.log(z)


def fpt(cs: NDArrayF64, K: int | None = None) -> NDArrayC128:
    k = K or len(cs) // 2
    qs = fit_q_poly(cs, k)
    ps = fit_p_poly(cs, qs, k)
    zs = poly_roots(qs)
    ds = resonant_amplitudes(cs, qs, ps, zs, k)
    return np.stack((ds, zs))


def bifpt(cs: NDArrayF64, K: int | None = None) -> NDArrayC128:
    ds_in_, zs_in_ = np.unstack(fpt(cs, K))
    ds_in = ds_in_[np.abs(zs_in_) < 1]
    zs_in = zs_in_[np.abs(zs_in_) < 1]

    ds_rev, zs_rev = np.unstack(fpt(np.flip(cs), K))
    ds_out_ = ds_rev * zs_rev ** (cs.size - 1)
    zs_out_ = 1 / zs_rev
    ds_out = ds_out_[np.abs(zs_out_) > 1]
    zs_out = zs_out_[np.abs(zs_out_) > 1]

    ds = np.concatenate((ds_in, ds_out))
    zs = np.concatenate((zs_in, zs_out))

    return np.stack((ds, zs))


def reconstruction(ds: NDArrayC128, zs: NDArrayC128, N: int) -> NDArrayF64:
    return np.real(np.vander(zs, N, increasing=True).T @ ds)
