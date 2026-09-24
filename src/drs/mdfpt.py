import numpy as np
import scipy.linalg as la


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
        return self.forward.signal(length) + self.backward.signal(length)

    @classmethod
    def fit(cls, signal, degree):
        n = signal.shape[0]
        ls_forward, xs_forward = LinearRecurrence.fit(signal, degree).eig()
        mask_forward = np.abs(ls_forward) < 1
        ls_forward = ls_forward[mask_forward]
        xs_forward = xs_forward[mask_forward]
        ls_backward, xs_backward = LinearRecurrence.fit(np.flipud(signal), degree).eig()
        mask_backward = np.abs(ls_backward) < 1
        ls_backward = ls_backward[mask_backward]
        xs_backward = xs_backward[mask_backward]

        vander_forward = np.vander(ls_forward, n, increasing=True).T
        vander_backward = np.vander(ls_backward, n, increasing=True).T
        vander = np.concatenate([vander_forward, np.flipud(vander_backward)], axis=1)
        result = la.lstsq(vander, signal)
        if result is not None:
            solution = result[0]
        else:
            raise la.LinAlgError
        ds_forward = solution[: len(ls_forward)]
        ds_backward = solution[len(ls_forward) :]

        forward = ResonanceSet(ds_forward, ls_forward, xs_forward, "forward")
        backward = ResonanceSet(ds_backward, ls_backward, xs_backward, "backward")
        return ResonanceBasis(forward, backward)


class ResonanceSet:
    def __init__(self, amplitudes, frequencies, vectors, direction):
        self.amplitudes = np.array(amplitudes)
        self.frequencies = np.array(frequencies)
        self.vectors = np.array(vectors)
        self.direction = direction
        assert self.amplitudes.shape[0] == self.frequencies.shape[0]
        self.cardinality = self.frequencies.shape[0]
        self.dimension = self.amplitudes.shape[1]

    def signal(self, length):
        vander = np.vander(self.frequencies, length, increasing=True).T
        signal = vander @ self.amplitudes
        if self.direction == "backward":
            return np.flipud(signal)
        else:
            return signal


class LinearRecurrence:
    def __init__(self, coefficients):
        self.coefficients = np.array(coefficients)
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
        hankel = np.stack([s[k : k + K] for k in range(N - K)]).reshape(N - K, M * K)
        target = s[K:]
        result = la.lstsq(hankel, target)
        if result is not None:
            solution = result[0]
            coefficients = solution.reshape(K, M, M).transpose(0, 2, 1)
            return LinearRecurrence(coefficients)
        else:
            raise la.LinAlgError

    def eig(self):
        k, m = self.degree, self.dimension
        c = np.zeros((m * k, m * k))
        c[:-m, m:] = np.diag(np.ones(m * (k - 1)))
        c[-m:, :] = self.coefficients.transpose(1, 0, 2).reshape(m, m * k)
        result = la.eig(c)
        e = result[0]
        vr = result[1][:m].T
        v = vr / la.norm(vr, axis=1)[:, np.newaxis]
        return e, v


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
        s = np.einsum("ijk,ik->j", self.recurrence.coefficients, self._tail)
        self._tail = np.concatenate([self._tail[1:], s])
        return s

    def repeat(self, length):
        return np.array([s for s, _ in zip(self, range(length))])
