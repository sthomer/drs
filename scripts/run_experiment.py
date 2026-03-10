from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np

from drs.config import load_config
from drs.io import from_wav, to_csv
from drs.preprocessing import preprocess_signal
from drs.analysis import compute_drs, compute_reconstruction
from drs.plotting import plot_drs, plot_signals


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    return config


def create_run(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = config.get("run_name", "run")
    run_dir = Path(config["paths"]["output_dir"]) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=2)
    return run_dir


def main():
    config = get_config()
    signal_raw, sample_rate = from_wav(config["paths"]["raw_data"])
    signal_preprocessed = preprocess_signal(signal_raw, **config["preprocessing"])
    spectrogram = compute_drs(signal_preprocessed, **config["analysis"])
    signal_reconstruction = compute_reconstruction(spectrogram)

    run_dir = create_run(config)
    np.save(run_dir / "signal_preprocessed.npy", signal_preprocessed)
    np.save(run_dir / "signal_reconstruction.npy", signal_reconstruction)
    to_csv(run_dir / "drs.csv", spectrogram, sample_rate)

    if config["plotting"]["plot_drs"]:
        plot_drs(run_dir / "drs.png", spectrogram, sample_rate)
    if config["plotting"]["plot_reconstruction"]:
        plot_signals(run_dir / "reconstruction.png", signal_preprocessed, signal_reconstruction)

    print(f"Saved run to {run_dir}")

if __name__ == "__main__":
    main()
