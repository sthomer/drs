import numpy as np
import scipy.linalg as la
import torch
import src.drs.recurrence as recurrence


def random_amplitudes(K, M):
    D_real = 2 * (np.random.rand(M * K // 2, M) - 0.5)
    D_imag = 2j * (np.random.rand(M * K // 2, M) - 0.5)
    D = D_real + D_imag
    D_conj = np.conj(D)
    Ds = np.block([[D], [D_conj]])
    return Ds


def random_frequencies(K, M):
    r = 1 - np.random.rand(M * K // 2) / 100
    theta = np.pi * np.random.rand(M * K // 2)
    L = r * np.exp(1j * theta)
    L_conj = np.conj(L)
    Ls = np.concatenate([L, L_conj])
    return Ls


# def test_frequency(K, M, N, runs=10):
#     matches = 0
#     mismatches = []
#     for n in range(runs):
#         Ds = random_amplitudes(K, M)
#         Ls = random_frequencies(K, M)
#         dirs = np.ones(M * K)
#         Rs = ResonanceBasis(Ds, Ls, dirs)
#         signal = np.real(Rs.signal(N))
#         Ls_, _ = polyeig_tensor(recurrence.fit_q_tensor_poly(signal, K))
#         LL = Ls.reshape(1, -1) - Ls_.reshape(-1, 1)
#         print(np.min(np.abs(LL), axis=1))
#         same = np.all(np.any(np.isclose(np.abs(LL), 0), axis=1))
#         if same:
#             matches += 1
#         # else:
#         #     print(np.any(np.isclose(LL, 0), axis=1))
#         #     mismatches.append((Ls, Ls_))
#     return matches / runs  # , mismatches
#
#
# def test_amplitudes(K, M, N, runs=10):
#     maes = []
#     for n in range(runs):
#         Ds = random_amplitudes(K, M)
#         Ls = random_frequencies(K, M)
#         dirs = np.ones(M * K)
#         Rs = ResonanceBasis(Ds, Ls, dirs)
#         signal = np.real(Rs.signal(N))
#         Ds_ = coeffs(Ls, signal)
#         maes.append(np.mean(np.abs(Ds - Ds_) / np.abs(Ds)))
#     return np.mean(maes)
#
#
# def test_reconstruction(K, M, N, runs=10):
#     maes = []
#     for n in range(runs):
#         Ds = random_amplitudes(K, M)
#         Ls = random_frequencies(K, M)
#         dirs = np.ones(M * K)
#         Rs = ResonanceBasis(Ds, Ls, dirs)
#         signal = np.real(Rs.signal(N))
#         Ls_, _ = polyeig_tensor(recurrence.fit_q_tensor_poly(signal, K))
#         Ds_ = coeffs(Ls_, signal)
#         Rs_ = ResonanceBasis(Ds_, Ls_, dirs)
#         signal_ = np.real(Rs_.signal(N))
#         maes.append(np.mean(np.abs(signal - signal_) / np.abs(signal)))
#     return np.mean(maes)


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
    wv = la.eig(C, right=True)
    E, V = wv[0], wv[1][:M]
    X = V / la.norm(V, axis=1, keepdims=True)
    return E, X


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
    D = la.lstsq(vander, signal)
    if D is not None:
        return D[0][: len(ls_forward)], D[0][len(ls_forward) :]
    else:
        raise la.LinAlgError


def fpt(signal, degree):
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
    qs_forward = recurrence.fit_q_tensor_poly(signal, degree)
    ls_forward, _ = polyeig_tensor(qs_forward)
    ls_forward = ls_forward[np.abs(ls_forward) < 1]
    qs_backward = recurrence.fit_q_tensor_poly(np.flipud(signal), degree)
    ls_backward, _ = polyeig_tensor(qs_backward)
    ls_backward = ls_backward[np.abs(ls_backward) < 1]
    ds_forward, ds_backward = coeffs(ls_forward, ls_backward, signal)
    return ResonanceBasis(ds_forward, ls_forward, ds_backward, ls_backward)
