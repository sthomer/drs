import numpy as np


def preprocess_signal(signal, sample_rate=44100, offset=0, length=None, mono=False):
    if mono and len(signal.shape) > 1:
        signal = signal[:, 0]
    if length is None:
        length = len(signal)
    return signal[offset : offset + length]


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
