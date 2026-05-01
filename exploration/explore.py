import numpy as np
import numpy.linalg as la
import drs

signal, sample_rate = drs.io.from_wav("data/raw/cello_drone.wav")
S = signal[1000:1030]
K = 10
Q = drs.core.fit_q_tensor_poly(S, K)
E, X = drs.core.polyeig_tensor(Q)
# D = drs.core.coeffs(E, X, S)
