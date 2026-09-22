"""
Smoke test: verifies the multimodal fusion model works end-to-end on
synthetic tensors (random MRI volumes + random EEG feature vectors),
without needing the real LEMON dataset downloaded.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import FusionModel

N_EEG = 5 * 32  # 5 bands x 32 channels


def test_fusion_mode():
    model = FusionModel(n_eeg_features=N_EEG, mode="fusion")

    mri = torch.randn(4, 1, 64, 64, 64)
    eeg = torch.randn(4, N_EEG)
    age = torch.rand(4) * 50 + 20

    pred = model(mri, eeg)
    assert pred.shape == (4,), f"unexpected shape {pred.shape}"

    # also check backward doesn't blow up, not just forward shape
    loss = torch.nn.functional.l1_loss(pred, age)
    loss.backward()
    print("fusion ok, shape", pred.shape)


def test_mri_only_mode():
    model = FusionModel(n_eeg_features=N_EEG, mode="mri_only")
    mri = torch.randn(2, 1, 64, 64, 64)
    eeg = torch.randn(2, N_EEG)  # still has to be passed even though it's ignored
    pred = model(mri, eeg)
    assert pred.shape == (2,)
    print("mri_only ok")


def test_eeg_only_mode():
    model = FusionModel(n_eeg_features=N_EEG, mode="eeg_only")
    mri = torch.randn(2, 1, 64, 64, 64)
    eeg = torch.randn(2, N_EEG)
    pred = model(mri, eeg)
    assert pred.shape == (2,)
    print("eeg_only ok")


if __name__ == "__main__":
    test_fusion_mode()
    test_mri_only_mode()
    test_eeg_only_mode()
    print("all good")