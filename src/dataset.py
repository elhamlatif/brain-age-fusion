"""
Multimodal dataset for brain-age prediction.

Each sample returns:
    - MRI volume          shape (1, D, H, W)
    - EEG band-power vec  shape (n_features,)
    - age                 float
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class MultimodalBrainDataset(Dataset):
    def __init__(self, subjects_csv=None, mri_dir=None, eeg_dir=None):
        self.subjects_csv = Path(subjects_csv) if subjects_csv else PROCESSED_DIR / "subjects.csv"
        self.mri_dir = Path(mri_dir) if mri_dir else PROCESSED_DIR / "mri"
        self.eeg_dir = Path(eeg_dir) if eeg_dir else PROCESSED_DIR / "eeg"

        self.subjects = pd.read_csv(self.subjects_csv)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        row = self.subjects.iloc[index]
        subject_id = row["subject_id"]
        age = float(row["age"])

        mri = np.load(self.mri_dir / f"{subject_id}.npy")
        eeg = np.load(self.eeg_dir / f"{subject_id}.npy")

        mri_tensor = torch.from_numpy(mri).unsqueeze(0)  # (1, D, H, W)
        eeg_tensor = torch.from_numpy(eeg)               # (n_features,)
        age_tensor = torch.tensor(age, dtype=torch.float32)

        return mri_tensor, eeg_tensor, age_tensor