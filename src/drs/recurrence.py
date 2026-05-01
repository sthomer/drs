import numpy as np
import scipy.linalg as la
import torch


def test(K, M, runs=10):
    maes = []
    for n in range(runs):
        Q = (np.random.rand(K, M, M) - 0.5) / 10
        S = (np.random.rand(K, M) - 0.5) / 10
        A = LinearRecurrence(Q).generate_from(S).repeat(M * K)
        signal = np.block([[S], [A]])
        Q_ = fit_q_tensor_poly(signal, K)
        maes.append(np.mean(np.abs(Q - Q_) / np.abs(Q)))
    return np.mean(maes)


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
    Q = la.lstsq(H, B)
    # Q = torch.linalg.lstsq(torch.from_numpy(H), torch.from_numpy(B))
    if Q is not None:
        return Q[0].reshape(K, M, M).transpose(0, 2, 1)
        # return Q[0].numpy().reshape(K, M, M).transpose(0, 2, 1)
    else:
        raise la.LinAlgError
