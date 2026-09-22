"""
Extract relative band-power features from resting-state (eyes-closed) EEG
using MNE-Python.

Output for each subject:
    data/processed/eeg/<subject_id>.npy
    shape: (n_channels * n_bands,)
"""

from pathlib import Path

import mne
import numpy as np

mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "eeg"

EEG_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

FREQ_MIN = 1
FREQ_MAX = 45
SUPPORTED_EXTENSIONS = (".set", ".vhdr", ".edf")
EPSILON = 1e-12

# LEMON has both eyes-open and eyes-closed resting EEG per subject.
# We only want eyes-closed, otherwise the second file would overwrite the first.
EEG_CONDITION = "eyesclosed"

def compute_band_power_features(raw):
    """
    Compute relative band-power features from a raw EEG recording.

    Returns a flat vector of shape (n_channels * n_bands,).
    """
    raw.filter(FREQ_MIN, FREQ_MAX, fir_design="firwin")

    psd, freqs = raw.compute_psd(
        fmin=FREQ_MIN, fmax=FREQ_MAX, method="welch"
    ).get_data(return_freqs=True)

    total_power = psd.sum(axis=1, keepdims=True)

    band_features = []
    for band_name, (fmin, fmax) in EEG_BANDS.items():
        in_band = (freqs >= fmin) & (freqs < fmax)
        band_power = psd[:, in_band].sum(axis=1, keepdims=True)
        relative_power = band_power / (total_power + EPSILON)
        band_features.append(relative_power)

    return np.concatenate(band_features, axis=1).flatten()

def find_eeg_files(raw_dir):
    """
    Find all eyes-closed resting-state EEG files under a BIDS-style directory.
    Explicitly filters out eyes-open files so we don't process both conditions.
    """
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = f"sub-*/**/eeg/*{EEG_CONDITION}*{ext}"
        files.extend(raw_dir.glob(pattern))
    return sorted(files)

def load_raw_eeg(fpath):
    """Load a raw EEG file based on its extension."""
    if fpath.suffix == ".set":
        return mne.io.read_raw_eeglab(fpath, preload=True)
    if fpath.suffix == ".vhdr":
        return mne.io.read_raw_brainvision(fpath, preload=True)
    if fpath.suffix == ".edf":
        return mne.io.read_raw_edf(fpath, preload=True)
    raise ValueError(f"Unsupported EEG file extension: {fpath}")

def extract_subject_id(fpath):
    """Pull the subject ID from the filename (e.g. sub-010001_...)."""
    return fpath.name.split("_")[0]

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eeg_files = find_eeg_files(RAW_DIR)
    if not eeg_files:
        print(f"No eyes-closed EEG files found under: {RAW_DIR}")
        return

    print(f"Found {len(eeg_files)} eyes-closed EEG files.")

    for fpath in eeg_files:
        subject_id = extract_subject_id(fpath)
        raw = load_raw_eeg(fpath)
        features = compute_band_power_features(raw)

        out_path = OUT_DIR / f"{subject_id}.npy"
        np.save(out_path, features.astype(np.float32))
        print(f"  {subject_id}: {features.shape} → {out_path}")

if __name__ == "__main__":
    run()