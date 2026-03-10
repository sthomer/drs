import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from drs.core import power, resonant_frequencies

def plot_signals(filename, signal_a, signal_b):
    plt.figure()
    plt.plot(signal_a)
    plt.plot(signal_b)
    plt.savefig(filename)


def plot_drs(filename, spectrogram, sample_rate, ylim=None):

    powers = sorted(
        [
            pow
            for dzs, _, _, window_size in spectrogram
            for pow in power(dzs, window_size)
        ]
        + [
            pow
            for _, dzs_rev, _, window_size in spectrogram
            for pow in power(dzs_rev, window_size)
        ]
    )
    vmin = powers[-len(powers) // 4]
    vmax = powers[-1]
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap("binary")

    plt.figure()
    for dzs, dzs_rev, offset, window_size in spectrogram:
        _, z = np.unstack(dzs)
        _, z_rev = np.unstack(dzs_rev)
        zs = np.concatenate((z, z_rev))
        frequencies = np.real(resonant_frequencies(zs, sample_rate))
        powers = np.concatenate((power(dzs, window_size), power(dzs_rev, window_size)))
        for frequency, pow in zip(frequencies, powers):
            plt.plot(
                [offset, offset + window_size],
                [frequency, frequency],
                color=cmap(norm(pow)),
            )
    if ylim is not None:
        plt.ylim(ylim)

    plt.savefig(filename)
