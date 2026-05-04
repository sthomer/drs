try:
    from . import _eiscor_ext
except ImportError as e:
    raise ImportError(
        "The compiled extension '_eiscor_ext' could not be imported. "
        "Reinstall the package with: python -m pip install --no-build-isolation -e ."
    ) from e

import numpy as np
import scipy.linalg as la
import torch


class ResonanceBasis:
    def __init__(self, ds_forward, ls_forward, ds_backward, ls_backward):
        self.ds_forward = np.array(ds_forward)  # shape(M*K, M)
        self.ls_forward = np.array(ls_forward)  # shape(M*K)
        self.ds_backward = np.array(ds_backward)  # shape(M*K, M)
        self.ls_backward = np.array(ls_backward)  # shape(M*K)
        self.dimension = self.ds_forward.shape[1]

    def signal(self, length):
        vander_forward = np.vander(np.array(self.ls_forward), length, increasing=True).T
        signal_forward = vander_forward @ self.ds_forward
        vander_backward = np.vander(
            np.array(self.ls_backward), length, increasing=True
        ).T
        signal_backward = vander_backward @ self.ds_backward
        signal = signal_forward + np.flipud(signal_backward)
        return signal


class LinearRecurrence:
    def __init__(self, coefficients):
        K, M, _ = coefficients.shape
        self.coefficients = coefficients
        self.degree = K
        self.dimension = M

    def generate_from(self, signal):
        return LinearRecurrenceGenerator(self, signal)


class LinearRecurrenceGenerator:
    def __init__(self, recurrence, signal):
        N, M = signal.shape
        assert N == recurrence.degree
        assert M == recurrence.dimension
        self.recurrence = recurrence
        self.signal = signal

    def __iter__(self):
        self._tail = self.signal[: self.recurrence.degree]
        return self

    def __next__(self):
        # s = sum(
        #     self.recurrence.coefficients[k] @ self._tail[k]
        #     for k in range(self.recurrence.degree)
        # )
        s = np.einsum("ijk,ik->j", self.recurrence.coefficients, self._tail)
        self._tail = np.block([[self._tail[1:]], [s]])
        return s

    def repeat(self, length):
        return np.array([s for s, _ in zip(self, range(length))])


def hankel_tensor(s, K):
    """
    Construct a Hankel tensor for a linear recurrence.

    Parameters
    ----------
    s : ndarray, shape(N, M)
        Sequence of vectors of length M
    K : int
        Order of the linear recurrence.

    Returns
    -------
    ndarray, shape(N-K, K, M)
        Hankel matrix
    """
    N = s.shape[0]
    return np.stack([s[k : k + K] for k in range(N - K)])


def fit_q_tensor_poly(s, K):
    """
    Solve for the coefficients of a linear recurrence.

    Parameters
    ----------
    s : ndarray, shape(N, M)
        Sequence of vectors of length M
    K : int
        Order of the linear recurrence

    Returns
    -------
    ndarray, shape(K, M, M)
        Matrix coefficients
    """
    N, M = s.shape
    H_ = hankel_tensor(s, K)
    H = H_.reshape(N - K, M * K)
    B = s[K:]
    # Q = la.lstsq(H, B)
    Q = torch.linalg.lstsq(torch.from_numpy(H), torch.from_numpy(B.copy()))
    if Q is not None:
        # return Q[0].reshape(K, M, M).transpose(0, 2, 1)
        return Q[0].numpy().reshape(K, M, M).transpose(0, 2, 1)
    else:
        raise la.LinAlgError


def companion_tensor(Q):
    """
    Construct the Frobenius companion tensor for a monic matrix polynomial.

    Parameters
    ----------
    Q : ndarray, shape(K, M, M)
        Matrix coefficients as a block "row vector"

    Returns
    -------
    ndarray, shape(K, K, M, M)
        Frobenius companion tensor
    """
    K, M, _ = Q.shape
    Z = np.zeros((M, M))
    I = np.eye(M)
    return np.stack([k * [Z] + [I] + (K - k - 1) * [Z] for k in range(1, K)] + [Q])


def polyeig_tensor(Q):
    """
    Solve polynomial eigenvalue problem for a monic matrix polynomial.

    Parameters
    ----------
    Q : ndarray, shape(K, M, M)
        Matrix coefficients as a block "row vector"

    Returns
    -------
    eigenvalue : ndarray, shape(M*K)
        Vector of eigenvalues
    eigenvectors : ndarray, shape(M, M*K)
        Matrix of normalized eigenvectors
    """
    K, M, _ = Q.shape
    C_ = companion_tensor(Q)
    C = C_.transpose(0, 2, 1, 3).reshape(M * K, M * K)
    E = torch.linalg.eigvals(torch.from_numpy(C))
    return E
    # wv = la.eig(C, right=True)
    # E, V = wv[0], wv[1][:M]
    # X = V / la.norm(V, axis=1, keepdims=True)
    # return E, X


def coeffs(ls_forward, ls_backward, signal):
    """
    Solve for resonance vectors amplitudes.

    Parameters
    ----------
    ls_forward : ndarray, shape(K_forward)
        Stable eigenvalues of the signal
    ls_backward : ndarray, shape(K_backward)
        Stable eigenvalues of the reverse signal
    signal : ndarray, shape(N, M)
        Signal of length N with M channels

    Returns
    -------
    ndarray, shape(K_forward, M)
        Forward resonance vector amplitudes
    ndarray, shape(K_backward, M)
        Backward resonance vector amplitudes

    Notes
    ----
    K_forward + K_backward = M * K
    """
    N, _ = signal.shape
    vander_forward = np.vander(ls_forward, N, increasing=True).T
    vander_backward = np.vander(ls_backward, N, increasing=True).T
    vander = np.block([[vander_forward, np.flipud(vander_backward)]])
    D = torch.linalg.lstsq(
        torch.from_numpy(vander), torch.from_numpy(signal.astype(np.complex128))
    )
    # D = la.lstsq(vander, signal)
    if D is not None:
        return D[0][: len(ls_forward)], D[0][len(ls_forward) :]
    else:
        raise la.LinAlgError


def mdfpt(signal, degree):
    """
    Decompose a signal into its resonance basis.

    If the signal has M channels and the resonance basis has degree K,
    there are M*K resonance vectors in the decomposition
    split between damping and ramping resonances.

    Parameters
    ----------
    signal : ndarray, shape(N, M)
        Signal of length N with M channels
    degree : int
        Degree of the underlying matrix polynomial

    Returns
    -------
    ResonanceBasis
        A resonance basis composed of:
        - Initial resonant amplitude vectors for damping resonances
        - Damping resonant frequencies (damping in forward time)
        - Final resonance amplitude vectors for ramping resonances
        - Ramping resonant frequencies (damping in reverse time)
    """
    qs_forward = fit_q_tensor_poly(signal, degree)
    ls_forward = polyeig_tensor(qs_forward)
    ls_forward = ls_forward[np.abs(ls_forward) < 1]
    qs_backward = fit_q_tensor_poly(np.flipud(signal), degree)
    ls_backward = polyeig_tensor(qs_backward)
    ls_backward = ls_backward[np.abs(ls_backward) < 1]
    ds_forward, ds_backward = coeffs(ls_forward, ls_backward, signal)
    return ResonanceBasis(ds_forward, ls_forward, ds_backward, ls_backward)


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


def inner_product(dzs_a, dzs_b, sample_rate):
    """
    Compute the inner product between two resonance spectra
    $\\langle \\sigma_a \\mid \\sigma_b \\rangle$,
    where both resonance spectra consist of
    only damping resonance or only ramping resonances.

    Parameters
    ----------
    dzs_a : ndarray, shape(2, J)
        Resonances of the resonance spectrum in the first argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_b : ndarray, shape(2, K)
        Resonances of the resonance spectrum in the second argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    sample_rate : float
        Sample rate associated with the resonance spectrum.

    Returns
    -------
    complex
        Inner product between the resonance spectra.

    See Also
    --------
    cosine_similarity
    cosine_distance

    Notes
    -----
    You must not compare damping resonances and ramping resonances!

    Examples
    --------
    Compute the resonance spectrum inner product
    between a signal and itself additive noise.

    >>> import numpy as np
    >>> from drs.io import from_wav
    >>> from drs.core import fpt, rs_ip
    >>> signal, sample_rate = from_wav("data/raw/zero.wav")
    >>> signal_a = signal[1024:2048]
    >>> signal_b = signal_a + np.random.normal(0, 0.01, len(signal_a))
    >>> dzs_a, dzs_rev_a = fpt(signal_a), fpt(np.flip(signal_a))
    >>> dzs_b, dzs_rev_b = fpt(signal_b), fpt(np.flip(signal_b))
    >>> ip_ab = rs_ip(dzs_a, dzs_b, dzs_rev_b, sample_rate)
    >>> ip_ab_rev = rs_ip(dzs_rev_a, dzs_rev_b, sample_rate)
    >>> ip_ab + ip_ab_rev
    """

    ds_a, zs_a = np.unstack(dzs_a)
    ws_a = resonant_frequencies(zs_a, sample_rate)
    dj = ds_a[:, np.newaxis]
    wj = ws_a[:, np.newaxis]

    ds_b, zs_b = np.unstack(dzs_b)
    ws_b = resonant_frequencies(zs_b, sample_rate)
    dk = ds_b[np.newaxis, :]
    wk = ws_b[np.newaxis, :]

    return 1j * np.sum((np.conj(dj) @ dk) / (np.conj(wj) - wk))


def cosine_similarity(dzs_a, dzs_rev_a, dzs_b, dzs_rev_b, sample_rate):
    """
    Compute the cosine similarity between two resonance spectra
    $s_c(\\sigma_a, \\sigma_b)$.

    Parameters
    ----------
    dzs_a : ndarray, shape(2, J_damping)
        Damping resonances of the resonance spectrum in the first argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_rev_a : ndarray, shape (2, J_ramping)
        Ramping resonances of the resonance spectrum in the first argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_a : ndarray, shape(2, K_damping)
        Damping resonances of the resonance spectrum in the second argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_rev_a : ndarray, shape (2, K_ramping)
        Ramping resonances of the resonance spectrum in the second argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    sample_rate : float
        Sample rate associated with the resonance spectrum.

    Returns
    -------
    float
        Cosine similarity ranging from -1 to 1.

    See Also
    --------
    inner_product
    cosine_distance

    Examples
    --------
    Compute the cosine similarity between a signal and itself additive noise.

    >>> import numpy as np
    >>> from drs.io import from_wav
    >>> from drs.core import fpt, rs_cos_sim
    >>> signal, sample_rate = from_wav("data/raw/zero.wav")
    >>> signal_a = signal[1024:2048]
    >>> signal_b = signal_a + np.random.normal(0, 0.01, len(signal_a))
    >>> dzs_a, dzs_rev_a = fpt(signal_a), fpt(np.flip(signal_a))
    >>> dzs_b, dzs_rev_b = fpt(signal_b), fpt(np.flip(signal_b))
    >>> cosine_similarity(dzs_a, dzs_rev_a, dzs_b, dzs_rev_b, sample_rate)
    """

    ip_ab = inner_product(dzs_a, dzs_b, sample_rate)
    ip_ab_rev = inner_product(dzs_rev_a, dzs_rev_b, sample_rate)
    top = np.real(ip_ab + ip_ab_rev)

    normsq_a = inner_product(dzs_a, dzs_a, sample_rate)
    normsq_a_rev = inner_product(dzs_rev_a, dzs_rev_a, sample_rate)
    normsq_b = inner_product(dzs_b, dzs_b, sample_rate)
    normsq_b_rev = inner_product(dzs_rev_b, dzs_rev_b, sample_rate)
    bot = np.sqrt(normsq_a + normsq_a_rev) * np.sqrt(normsq_b + normsq_b_rev)

    return np.real(top / bot)


def cosine_distance(dzs_a, dzs_rev_a, dzs_b, dzs_rev_b, sample_rate):
    """
    Compute the cosine distance between two resonance spectra
    $d_c(\\sigma_a, \\sigma_b)$.

    Parameters
    ----------
    dzs_a : ndarray, shape(2, J_damping)
        Damping resonances of the resonance spectrum in the first argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_rev_a : ndarray, shape (2, J_ramping)
        Ramping resonances of the resonance spectrum in the first argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_a : ndarray, shape(2, K_damping)
        Damping resonances of the resonance spectrum in the second argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    dzs_rev_a : ndarray, shape (2, K_ramping)
        Ramping resonances of the resonance spectrum in the second argument,
        where the first row contains the resonant amplitudes
        and the second row contains the correspond poles.
    sample_rate : float
        Sample rate associated with the resonance spectrum.

    Returns
    -------
    float
        Cosine distance ranging from 0 to 2.

    See Also
    --------
    inner_product
    cosine_similarity

    Examples
    --------
    >>> import numpy as np
    >>> from drs.io import from_wav
    >>> from drs.core import fpt, rs_cos_dist
    >>> signal, sample_rate = from_wav("data/raw/zero.wav")
    >>> signal_a = signal[1024:2048]
    >>> signal_b = signal_a + np.random.normal(0, 0.01, len(signal_a))
    >>> dzs_a, dzs_rev_a = fpt(signal_a), fpt(np.flip(signal_a))
    >>> dzs_b, dzs_rev_b = fpt(signal_b), fpt(np.flip(signal_b))
    >>> cosine_distance(dzs_a, dzs_rev_a, dzs_b, dzs_rev_b, sample_rate)
    """

    sim = cosine_similarity(dzs_a, dzs_rev_a, dzs_b, dzs_rev_b, sample_rate)
    return np.sqrt(2 * (1 - sim))
