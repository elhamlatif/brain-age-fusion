"""
Load structural T1 MRI volumes (BIDS layout) and resample them to a fixed
small resolution so they can fit into a lightweight 3D CNN.

Important:
    This script assumes the volumes are already skull-stripped / brain-extracted.
    The raw LEMON T1s are not. If you haven't run BET (or equivalent) yet, do it first:

        bet sub-XX_T1w.nii.gz sub-XX_T1w_brain.nii.gz -f 0.5 -g 0

    Then point this script at the brain-extracted files (or rename them so the
    glob below still picks them up).

Output for each subject:
    data/processed/mri/<subject_id>.npy
    shape: (64, 64, 64)
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "mri"
TARGET_SHAPE = (64, 64, 64)


def resample_volume(volume, target_shape):
    """Resample a 3D volume to the target shape with linear interpolation."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def normalize(volume):
    """
    Robust min-max normalization.
    Clips the 1st–99th percentiles first so extreme outliers don't dominate.
    """
    volume = volume.astype(np.float32)
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume


def find_t1_files(raw_dir):
    """Find all T1w NIfTI files under a BIDS-style directory."""
    return sorted(raw_dir.glob("sub-*/**/*T1w.nii.gz"))


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1_files = find_t1_files(RAW_DIR)

    if not t1_files:
        print(f"No T1w files found under: {RAW_DIR}")
        return

    print(f"Found {len(t1_files)} T1w volumes.")

    for fpath in t1_files:
        subject_id = fpath.name.split("_")[0]
        img = nib.load(fpath)
        volume = img.get_fdata()

        volume = resample_volume(volume, TARGET_SHAPE)
        volume = normalize(volume)

        out_path = OUT_DIR / f"{subject_id}.npy"
        np.save(out_path, volume.astype(np.float32))
        print(f"  {subject_id}: {volume.shape} → {out_path}")


if __name__ == "__main__":
    run()