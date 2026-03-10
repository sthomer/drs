from drs.core import drs, reconstruction

def compute_drs(signal, window_size, step_size, degree=None):
    if degree is None:
        degree = window_size // 2
    spectrogram = drs(signal, window_size, step_size)
    return spectrogram

def compute_reconstruction(spectrogram):
    recon = []
    for dzs, dzs_rev, offset, window_size in spectrogram:
        r = reconstruction(dzs, dzs_rev, window_size)
        recon += r.tolist()
    return recon
