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

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
torch.set_default_device(device)


class ResonanceBasis:
    """
    A multidimensional resonance basis is composed of two sets of resonances
    a forward set and a backward set,

    Attributes
    ----------
    forward : ResonanceSet
        Forward resonance set.
        Resonant amplitudes are the initial amplitudes of each resonance.
        Resonance frequencies are damping.
    backward : ResonanceSet
        Backward resonance set.
        Resonant amplitudes are the final amplitudes of each resonance.
        Resonance frequencies are ramping. (Damping in reverse time)
    cardinality : int
        Total number of resonances in the resonance basis.
    dimension : int
        Dimension of resonance basis vectors

    Methods
    -------
    signal(length)
        Reconstruct a signal of a given length using this resonance basis.
    """

    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward
        self.cardinality = self.forward.cardinality + self.backward.cardinality
        assert self.forward.dimension == self.backward.dimension
        self.dimension = self.forward.dimension

    def signal(self, length):
        s = self.forward.signal(length) + self.backward.signal(length)
        return s.detach().cpu().numpy()

    @classmethod
    def fit(cls, signal, degree):
        signal = signal.to(device)
        n = signal.shape[0]
        ls_forward = LinearRecurrence.fit(signal, degree).eigvals()
        ls_forward = ls_forward[torch.abs(ls_forward) < 1]
        ls_backward = LinearRecurrence.fit(torch.flipud(signal), degree).eigvals()
        ls_backward = ls_backward[torch.abs(ls_backward) < 1]

        vander_forward = torch.vander(ls_forward, n, increasing=True).T
        vander_backward = torch.vander(ls_backward, n, increasing=True).T
        vander = torch.cat([vander_forward, torch.flipud(vander_backward)], 1)
        solution = torch.linalg.lstsq(vander, signal.to(torch.complex64)).solution
        ds_forward = solution[: len(ls_forward)]
        ds_backward = solution[len(ls_forward) :]
        forward = ResonanceSet(ds_forward, ls_forward, "forward")
        backward = ResonanceSet(ds_backward, ls_backward, "backward")
        return ResonanceBasis(forward, backward)


class ResonanceSet:
    def __init__(self, amplitudes, frequencies, direction):
        self.amplitudes = torch.tensor(amplitudes)
        self.frequencies = torch.tensor(frequencies)
        self.direction = direction
        assert self.amplitudes.shape[0] == self.frequencies.shape[0]
        self.cardinality = self.frequencies.shape[0]
        self.dimension = self.amplitudes.shape[1]

    def signal(self, length):
        vander = torch.vander(self.frequencies, length, increasing=True).T
        signal = vander @ self.amplitudes
        if self.direction == "backward":
            return torch.flipud(signal)
        else:
            return signal


class LinearRecurrence:
    def __init__(self, coefficients):
        self.coefficients = torch.tensor(coefficients)
        K, M, _ = self.coefficients.shape
        self.degree = K
        self.dimension = M

    def generate_from(self, signal):
        return LinearRecurrenceGenerator(self, signal)

    @classmethod
    def fit(cls, s, K):
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
        hankel = torch.stack([s[k : k + K] for k in range(N - K)]).reshape(N - K, M * K)
        target = s[K:]
        solution = torch.linalg.lstsq(hankel, target).solution
        coefficients = solution.reshape(K, M, M).transpose(2, 1)
        return LinearRecurrence(coefficients)

    def eigvals(self):
        k, m = self.degree, self.dimension
        c = torch.zeros((m * k, m * k))
        c[:-m, m:] = torch.diag(torch.ones(m * (k - 1)))
        c[-m:, :] = self.coefficients.transpose(0, 1).reshape(m, m * k)
        return torch.linalg.eigvals(c)


class LinearRecurrenceGenerator:
    def __init__(self, recurrence, signal):
        self.recurrence = recurrence
        self.signal = signal
        N, M = signal.shape
        assert N == recurrence.degree
        assert M == recurrence.dimension

    def __iter__(self):
        self._tail = self.signal[: self.recurrence.degree]
        return self

    def __next__(self):
        s = torch.einsum("ijk,ik->j", self.recurrence.coefficients, self._tail)
        self._tail = torch.cat([self._tail[1:], s])
        return s

    def repeat(self, length):
        return torch.tensor([s for s, _ in zip(self, range(length))])


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
