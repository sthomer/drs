from resonance import ResonanceSet
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def threedee(spectrogram: ResonanceSet):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    onsets, frequencies, powers = np.array(list(zip(*[
        (onset, frequency, power)
        for onset, spectrum in zip(spectrogram.onsets, spectrogram.spectra)
        for frequency, power in zip(spectrum.frequencies, spectrum.powers)
        if frequency >= 0
    ])))
    ax.scatter(onsets / spectrogram.sample_rate, frequencies, powers, c=powers)
    ax.set_ylim(0, 7000)


def plot_drs(spectrogram, signal=None, max_freq=3000, params=None):
    # if params is not None:
    #     title = ' '.join([(f'{k}={v}') for k, v in params.items()])
    # else:
    #     title = 'Resonance Spectrogram'
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((0, max_freq))
    # ax.set_title(title)
    spectrogram = spectrogram[np.isfinite(spectrogram.power)]
    normed = spectrogram.power / np.nanmax(spectrogram.power)
    # unit = fig.bbox.width * fig.bbox.height / 10000
    unit = 1280 * 960 / 20000

    ax.scatter(
        spectrogram.onsets / spectrogram.sample_rate,
        spectrogram.frequency,
        c=spectrogram.power,
        cmap="Greys",
        edgecolors='k'
    )

    def on_resize(event):
        ax.clear()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim((0, max_freq))
        # ax.set_title(title)
        unit = event.width * event.height / 20000
        ax.scatter(
            spectrogram.onsets / spectrogram.sample_rate,
            spectrogram.frequency,
            c=spectrogram.power,
            cmap="Greys",
            edgecolors='k'
        )

    ax.figure.canvas.mpl_connect('resize_event', on_resize)


def discrete(spectrogram, signal=None, max_freq=3000, params=None):
    # if params is not None:
    #     title = ' '.join([(f'{k}={v}') for k, v in params.items()])
    # else:
    #     title = 'Resonance Spectrogram'
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((0, max_freq))
    # ax.set_title(title)
    spectrogram = spectrogram[np.isfinite(spectrogram.power)]
    normed = spectrogram.power / np.nanmax(spectrogram.power)
    # unit = fig.bbox.width * fig.bbox.height / 10000
    unit = 1280 * 960 / 20000

    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, c=spectrogram.power,
               s=unit * normed)

    def on_resize(event):
        ax.clear()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim((0, max_freq))
        # ax.set_title(title)
        unit = event.width * event.height / 20000
        ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, c=spectrogram.power,
                   s=unit * normed)

    ax.figure.canvas.mpl_connect('resize_event', on_resize)


def discrete_old(spectrogram: ResonanceSet, mode: str = "power"):
    value_fn = {
        "power": lambda spectrum: spectrum.power,
        "amplitude": lambda spectrum: spectrum.amplitude,
        "harmonic": lambda spectrum: spectrum.harmonic_weight,
        "height": lambda spectrum: spectrum.height,
        "area": lambda spectrum: spectrum.area,
    }[mode]

    onsets, frequencies, values = np.array(list(zip(*[
        (onset, frequency, value)
        for onset, spectrum in zip(spectrogram.onsets, spectrogram.spectra)
        for frequency, value in zip(spectrum.frequency, value_fn(spectrum))
        if frequency >= 0
    ])))
    fig, ax = plt.subplots()

    duration = spectrogram.max_duration / spectrogram.sample_rate
    onsets = onsets / spectrogram.sample_rate
    mask = np.isfinite(values)
    onsets, frequencies, values = onsets[mask], frequencies[mask], values[mask]
    normed_values = values / np.max(values)
    # unit = fig.bbox.width * fig.bbox.height / 10000
    unit = 1280 * 960 / 10000
    ax.scatter(onsets, frequencies, c=values, s=unit * normed_values)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 5000)
    ax.set_title(f"Discrete Resonance Spectrogram (Mode: {mode})")

    def on_resize(event):
        ax.clear()
        unit = event.width * event.height / 10000
        ax.scatter(onsets, frequencies, c=values, s=unit * normed_values)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 5000)

    ax.figure.canvas.mpl_connect('resize_event', on_resize)


def continuous(spectrogram: ResonanceSet):
    onsets = spectrogram.onsets / spectrogram.sample_rate
    frequencies = np.linspace(0, 10000, 100)
    bins = [[spectrum.bin(f0, f1) for spectrum in spectrogram]
            for f0, f1 in zip(frequencies, frequencies[1:])]
    plt.contourf(onsets, frequencies[:-1], np.log(bins),
                 cmap='viridis', levels=100)
    plt.xlim(0, np.max(onsets))
    plt.ylim(0, np.max(frequencies))
    plt.title('Continuous Resonance Spectrogram')


def power_spectrum(spectrum: ResonanceSet, label=None):
    if len(spectrum) == 0:
        return
    frequencies = np.linspace(0, 5000, 10000)
    plt.plot(frequencies,
             np.log(np.abs(np.sum(spectrum.map(lambda _, res: res.at(frequencies)), axis=0) ** 2)),
             label=label)
    plt.title('Power Spectrum')

# def super_drs(rss, os, max_freq=3000):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#
#     cmap = mpl.cm.get_cmap("viridis")
#     N = 1000
#     rss_os = list(zip(np.flip(rss), np.flip(os)))
#     total = len(rss_os)
#     fig, axs = plt.subplots(len(rss), 1, frameon=False, sharex=True)
#     frequencies = np.linspace(0, max_freq, N)
#     powerss = []
#     min_powers = []
#     max_powers = []
#     for n, (resonances, onset) in enumerate(rss_os):
#         spectrum = ResonanceSet(resonances, [onset] * len(resonances))
#         powers = np.abs(np.sum(spectrum.map(lambda _, res: res.at(frequencies)), axis=0))**2
#         powerss.append(powers)
#         max_powers.append(np.max(powers))
#         min_powers.append(np.min(powers))
#     min_power = min(min_powers)
#     max_power = max(max_powers)
#     for n, powers in enumerate(powerss):
#         vals = powers / max_power
#         axs[n].plot(frequencies, vals, color='k', linewidth=0.75, zorder=len(rss) - n)
#         axs[n].set_yscale('log')
#         axs[n].set_ylim(min_power, max_power)
#         color = cmap(n / total)
#         axs[n].fill_between(frequencies, vals, color=color, zorder=total - n, alpha=0.75)
#     plt.tight_layout()
#     plt.show()


def super_drs(rss, os, max_freq=3000):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cmap = mpl.cm.get_cmap("viridis")
    N = 1000
    rss_os = list(zip(np.flip(rss), np.flip(os)))
    total = len(rss_os)
    ax = plt.subplot(frameon=False)
    max_powers = []
    frequencies = np.linspace(0, max_freq, N)
    powerss = []
    for n, (resonances, onset) in enumerate(rss_os):
        spectrum = ResonanceSet(resonances, [onset] * len(resonances))
        powers = np.abs(np.sum(spectrum.map(lambda _, res: res.at(frequencies)), axis=0))**2
        powerss.append(powers)
        max_powers.append(np.max(powers))
    max_power = max(max_powers)
    for n, powers in enumerate(powerss):
        vals = np.log1p(powers / max_powers[n])
        ax.plot(frequencies, 3 * vals + n, color='k', linewidth=0.75, zorder=len(rss) - n)
        color = cmap(n / total)
        ax.fill_between(frequencies, 3 * vals + n, n, color=color, zorder=total - n, alpha=0.75)
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", 1, pad=-0.1, sharey=ax, frame_on=False)
    ax2.barh(np.arange(len(max_powers)), np.log1p(max_powers))
    ax2.yaxis.set_tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    plt.tight_layout()
    plt.show()

def rougier():
    def curve():
        n = np.random.randint(1, 5)
        centers = np.random.normal(0.0, 1.0, n)
        widths = np.random.uniform(5.0, 50.0, n)
        widths = 10 * widths / widths.sum()
        scales = np.random.uniform(0.1, 1.0, n)
        scales /= scales.sum()
        X = np.zeros(500)
        x = np.linspace(-3, 3, len(X))
        for center, width, scale in zip(centers, widths, scales):
            X = X + scale * np.exp(-(x - center) * (x - center) * width)
        return X


    np.random.seed(123)
    cmap = mpl.cm.get_cmap("Spectral")
    fig = plt.figure(figsize=(8, 8))


    ax = None
    for n in range(3):
        ax = plt.subplot(1, 3, n + 1, frameon=False, sharex=ax)
        for i in range(50):
            Y = curve()
            X = np.linspace(-3, 3, len(Y))
            ax.plot(X, 3 * Y + i, color="k", linewidth=0.75, zorder=100 - i)
            color = cmap(i / 50)
            ax.fill_between(X, 3 * Y + i, i, color=color, zorder=100 - i)

            # Some random text on the right of the curve
            v = np.random.uniform(0, 1)
            if v < 0.4:
                text = "*"
                if v < 0.05:
                    text = "***"
                elif v < 0.2:
                    text = "**"
                ax.text(
                    3.0,
                    i,
                    text,
                    ha="right",
                    va="baseline",
                    size=8,
                    transform=ax.transData,
                    zorder=300,
                )

        ax.yaxis.set_tick_params(tick1On=False)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 53)
        ax.axvline(0.0, ls="--", lw=0.75, color="black", zorder=250)
        ax.text(
            0.0,
            1.0,
            "Value %d" % (n + 1),
            ha="left",
            va="top",
            weight="bold",
            transform=ax.transAxes,
        )

        if n == 0:
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks(np.arange(50))
            ax.set_yticklabels(["Serie %d" % i for i in range(1, 51)])
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(6)
                tick.label.set_verticalalignment("bottom")
        else:
            ax.yaxis.set_tick_params(labelleft=False)


    plt.tight_layout()
    plt.savefig("../../figures/anatomy/zorder-plots.png", dpi=600)
    plt.savefig("../../figures/anatomy/zorder-plots.pdf")
    plt.show()