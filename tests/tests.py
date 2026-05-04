# def test_recurrence(K, M, runs=10):
#     maes = []
#     for n in range(runs):
#         Q = (np.random.rand(K, M, M) - 0.5) / 10
#         S = (np.random.rand(K, M) - 0.5) / 10
#         A = LinearRecurrence(Q).generate_from(S).repeat(M * K)
#         signal = np.block([[S], [A]])
#         Q_ = fit_q_tensor_poly(signal, K)
#         maes.append(np.mean(np.abs(Q - Q_) / np.abs(Q)))
#     return np.mean(maes)
#
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
