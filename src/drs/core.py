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
    """
    Construct a Hankel matrix for a linear recurrence.

    Parameters
    ----------
    s : ndarray, shape(N, M)
        Sequence of vectors of length M
    K : int
        Order of the linear recurrence.

    Returns
    -------
    ndarray, shape(N - K, M * K)
        Hankel matrix
    """
    N, M = s.shape
    return np.block([[s[k : k + K].ravel()] for k in range(N-K)])


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


def fit_q_matpoly(s, K):
    """
    Solve for the coefficients of a linear recurrence.

    Parameters
    ----------
    s : ndarray, shape(M, N)
        Sequence of vectors of length M
    K : int
        Order of the linear recurrence

    Returns
    -------
    ndarray, shape(M, M*K)
        Matrix coefficients as a block "row vector"
    """
    A = hankel(s, K)
    b = s[K:]
    Qs = la.lstsq(A, b)
    if Qs is not None:
        return Qs[0].T
    else:
        raise la.LinAlgError

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
        Matrix coefficients as a block "row vector"
    """
    N, M = s.shape
    H_ = hankel_tensor(s, K)
    H = H_.reshape(N-K, M*K)
    B = s[K:]
    Q = la.lstsq(H, B)
    if Q is not None:
        return Q[0].reshape(K, M, M).transpose(0, 2, 1)
    else:
        raise la.LinAlgError


def companion(Q, K, M):
    """
    Construct the Frobenius companion matrix for a monic matrix polynomial.

    Parameters
    ----------
    Q : ndarray, shape(M, M*K)
        Matrix coefficients as a block "row vector"
    K : int
        Order of the linear recurrence
    M : int
        Dimension of vector signal

    Returns
    -------
    ndarray, shape(M*K, M*K)
        Frobenius companion matrix
    """
    return np.block([[np.zeros((M * (K - 1), M)), np.eye(M * (K - 1))], [Q]])

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
    return np.stack([k*[Z] + [I] + (K-k-1)*[Z] for k in range(1,K)] + [Q])


def polyeig(Q, K, M):
    """
    Solve polynomial eigenvalue problem for a monic matrix polynomial.

    Parameters
    ----------
    Q : ndarray, shape(M, M*K)
        Matrix coefficients as a block "row vector"
    K : int
        Order of the linear recurrence
    M : int
        Dimension of vector signal

    Returns
    -------
    eigenvalue : ndarray, shape(M*K)
        Vector of eigenvalues
    eigenvectors : ndarray, shape(M, M*K)
        Matrix of eigenvectors
    """
    wv = la.eig(companion(Q, K, M), right=True)
    return wv[0], wv[1][:M]

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
    C = C_.transpose(0,2,1,3).reshape(M*K, M*K)
    wv = la.eig(C, right=True)
    E, V = wv[0], wv[1][:M].T
    X = V / la.norm(V, axis=1, keepdims=True)
    return E, X
  

def coeffs(evs, s):
    """
    Solve for resonance vectors amplitudes.

    Parameters
    ----------
    evs : ndarray, shape(M*K)
        eigenvalues
    s : ndarray, shape(N, M)
        signal

    Returns
    -------
    ndarray, shape(M*K, M)
        Resonance vector amplitudes
    """
    N, M = s.shape
    D = la.lstsq(np.vander(evs, N, increasing=True).T, s)
    if D is not None:
        return D[0]
    else:
        raise la.LinAlgError


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
