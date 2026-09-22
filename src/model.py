"""
Fusion model for brain-age prediction.

MRI branch is a small 3D CNN, EEG branch is just an MLP over band powers.
Late fusion by concatenation. Also supports single-modality runs so we
can compare against the fusion setup in the ablation table.
"""

import torch
import torch.nn as nn

# embedding sizes -- tuned these by hand on a small val split, don't have
# a good reason for exactly these numbers but they worked fine
MRI_EMB = 64
EEG_EMB = 32

class MRIBranch(nn.Module):
    """3D CNN over (B, 1, D, H, W) -> (B, MRI_EMB)."""

    def __init__(self, embed_dim=MRI_EMB):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))

class EEGBranch(nn.Module):
    """MLP over (B, n_features) -> (B, EEG_EMB)."""

    def __init__(self, n_features, embed_dim=EEG_EMB):
        super().__init__()
        # 0.3 dropout seemed to help a bit; 0.5 was too much, killed the signal
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class FusionModel(nn.Module):
    """
    Concatenation-based late fusion.

    mode:
        "fusion"   -- both branches, concat, then regress
        "mri_only" -- ablation, MRI only
        "eeg_only" -- ablation, EEG only
    """

    MODES = ("fusion", "mri_only", "eeg_only")

    def __init__(self, n_eeg_features, mode="fusion",
                 mri_embed_dim=MRI_EMB, eeg_embed_dim=EEG_EMB):
        super().__init__()

        # fail loudly, I've wasted too much time on silent mode typos
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {self.MODES}")

        self.mode = mode

        if mode != "eeg_only":
            self.mri_branch = MRIBranch(embed_dim=mri_embed_dim)
        if mode != "mri_only":
            self.eeg_branch = EEGBranch(n_eeg_features, embed_dim=eeg_embed_dim)

        # TODO: try attention-based fusion at some point, cat is probably
        # leaving something on the table here
        if mode == "fusion":
            head_in = mri_embed_dim + eeg_embed_dim
        elif mode == "mri_only":
            head_in = mri_embed_dim
        else:
            head_in = eeg_embed_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, mri, eeg):
        # note: unused input is simply ignored, caller still has to pass something
        if self.mode == "fusion":
            x = torch.cat([self.mri_branch(mri), self.eeg_branch(eeg)], dim=1)
        elif self.mode == "mri_only":
            x = self.mri_branch(mri)
        else:
            x = self.eeg_branch(eeg)

        return self.head(x).squeeze(-1)

if __name__ == "__main__":
    # quick shape check, mostly to make sure I didn't mix up the einsum-free
    # flatten somewhere. 5 bands x 32 channels = 160, matches extract_eeg_features.
    n_eeg = 5 * 32
    mri = torch.randn(4, 1, 64, 64, 64)
    eeg = torch.randn(4, n_eeg)

    for mode in FusionModel.MODES:
        out = FusionModel(n_eeg, mode=mode)(mri, eeg)
        print(mode, out.shape)  # expect (4,)