from time import perf_counter

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

import serialization as ser
import visualization as viz
from fpt import FptConfig, drs
from utilities_store import plot_spectrogram_arrows

plt.rcParams["font.family"] = "Helvetica"

if __name__ == '__main__':
    n0 = 1024
    N = 1024
    # dir = "./data/input/london/violin"
    # filename = "violin_As5_long_forte_molto-vibrato"
    dir = "./data/input"
    # filename = "bach"
    filename = "zero"
    start = perf_counter()
    source, sample_rate = ser.from_file(filename, dir=dir)
    print(perf_counter() - start)
    signal = source #/ source[0] # source[n0:n0+N] / source[0] #[len(source)//2:len(source)//2+N]
    step_size = N // 2
    config = FptConfig(
        length=N,
        degree=N // 2,
        delay=0,
        sample_rate=sample_rate,
        power_threshold=0,
        decay_threshold=0
    )
    run = True
    if run:
        start = perf_counter()
        spectrogram = drs(signal, config, step_size)
        print(perf_counter() - start)
        ser.save(spectrogram)
    spectrogram = ser.load()
    recon = spectrogram.reconstruction
    # recon = np.sum(spectrogram.reconstruction, axis=0)
    plt.plot(signal)
    plt.plot(recon)
    plt.show()
    # ser.to_wav(recon, filename, sample_rate)

    # viz.super_drs(resonances, onsets, max_freq=5000)
    # viz.fpl_plot(spectra, onsets, sample_rate=sample_rate)
    # viz.plot_drs(spectrogram, max_freq=5000)

    # fig, ax = plt.subplots()
    # ts = np.arange(len(signal)) / sample_rate
    # ax.plot(ts, signal, label="Original")
    # ts_ = np.arange(len(recon)) / sample_rate
    # plt.plot(ts_, spectrogram.reconstruction, label='Reconstruction', alpha=0.5)
    # plt.legend()
    # ax.set_xlim(ts[0], ts[-1])
    # ax.set_xlabel("Time (s)")
    # y_max = np.max(abs(signal)) * 1.15
    # ax.set_ylim(-y_max, y_max)
    # ax.set_ylabel("Amplitude")
    # fig.set_figwidth(10)
    # fig.set_figheight(10)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.grid(axis="y")
    # plt.savefig(f'plots/signal_{filename}.svg', format='svg', bbox_inches="tight")
    # plt.show()

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    spectrum, freqs, t, im = plt.specgram(x=signal, Fs=sample_rate, NFFT=N, noverlap=N - step_size,
                                          window=plt.mlab.window_none)
    vmin = np.sort(spectrum.flatten())[-len(spectrum.flatten())//4]
    vmax = np.max(spectrum)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    idx = min(np.count_nonzero(freqs <= 5000), len(freqs))
    spectrum = spectrum[:idx]
    freqs = freqs[:idx]
    t = np.insert(t, 0, 0)
    im = ax.imshow(np.flipud(spectrum), aspect='auto',
                   extent=(np.amin(t) * sample_rate, np.amax(t) * sample_rate, freqs[0], freqs[-1]),
                   cmap='binary', norm=norm,
                   interpolation=None)
    # fig.colorbar(im, ax=ax, shrink=0.35, label="Power (dB)")
    plt.yticks([0, 1000, 2000, 3000, 4000, 5000], ['0', '1K', '2K', '3K', '4K', '5K'])
    txs = np.arange(0, t[-1], 0.1)
    plt.xticks(txs * sample_rate, [f'{tx:0.1f}' for tx in txs])
    ax.grid(axis="y")
    fig.set_figwidth(10)
    fig.set_figheight(6)
    plt.savefig(f'plots/fourier_{filename}.pdf', format='pdf', bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((0, 5000))
    # ax.set_title(title)
    spectrogram = spectrogram[np.isfinite(spectrogram.power)]
    power = spectrogram.power
    vmin = sorted(power)[-len(power)//4]
    vmax = np.max(power)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency,
               cmap='binary', norm=norm, c=power, marker='_')
    plt.yticks([0, 1000, 2000, 3000, 4000, 5000], ['0', '1K', '2K', '3K', '4K', '5K'])
    ax.set_xmargin(0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis="y")
    fig.set_figwidth(10)
    fig.set_figheight(6)
    plt.savefig(f'plots/discrete_{filename}.pdf', format='pdf', bbox_inches="tight")
    plt.show()
