import csv
import soundfile as sf
import numpy as np

from drs.core import spectral_params

from pathlib import Path
import json


def load_array(path: str) -> np.ndarray:
    return np.load(path)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def from_wav(filename):
    signal, sample_rate = sf.read(filename)
    return signal, sample_rate


def to_wav(signal, filename="temp", sample_rate=44100, dir="./data/output"):
    sf.write(f"{dir}/{filename}.wav", signal.real, sample_rate)


def to_csv(filename, spectrogram, sample_rate):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow((
            "onset",
            "duration",
            "sample_rate",
            "amplitude",
            "phase",
            "frequency",
            "decay",
        ))

        for dzs, dzs_rev, offset, window_size in spectrogram:
            params = spectral_params(dzs, dzs_rev, window_size, sample_rate)
            for amplitude, phase, frequency, decay in zip(*np.unstack(params)):
                writer.writerow((
                    offset,
                    window_size,
                    sample_rate,
                    amplitude,
                    phase,
                    frequency,
                    decay,
                ))
