"""
Evaluate a trained brain-age model on the held-out validation split.

Usage:
    python src/evaluate.py --model results/best_model_fusion.pt --mode fusion
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import MultimodalBrainDataset
from model import FusionModel

# Keep these in sync with train.py
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
VAL_FRACTION = 0.2
SEED = 42
BATCH_SIZE = 16


def build_validation_loader():
    """Return a DataLoader for the validation subset + number of EEG features."""
    full_dataset = MultimodalBrainDataset()
    n_eeg_features = full_dataset[0][1].shape[0]

    all_indices = list(range(len(full_dataset)))
    _, val_indices = train_test_split(
        all_indices,
        test_size=VAL_FRACTION,
        random_state=SEED,
    )

    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    return val_loader, n_eeg_features


def load_trained_model(model_path, n_eeg_features, mode, device):
    """Load a trained FusionModel and put it in eval mode."""
    model = FusionModel(n_eeg_features=n_eeg_features, mode=mode).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_predictions(targets, predictions, mae, r2, mode, out_path):
    """Scatter plot of chronological age vs predicted brain age."""
    plt.figure(figsize=(5, 5))
    plt.scatter(targets, predictions, alpha=0.6)

    lower = min(min(targets), min(predictions))
    upper = max(max(targets), max(predictions))
    plt.plot([lower, upper], [lower, upper], "r--", label="Ideal (y = x)")

    plt.xlabel("Chronological age")
    plt.ylabel("Predicted brain age")
    plt.title(f"Brain Age Prediction ({mode})\nMAE = {mae:.2f}, R² = {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a brain-age model on the validation set."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=["fusion", "mri_only", "eeg_only"],
        default="fusion",
        help="Which input mode the model was trained with",
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_loader, n_eeg_features = build_validation_loader()
    model = load_trained_model(args.model, n_eeg_features, args.mode, device)

    predictions, targets = [], []

    with torch.no_grad():
        for mri, eeg, age in val_loader:
            mri = mri.to(device)
            eeg = eeg.to(device)
            output = model(mri, eeg)
            predictions.extend(output.cpu().numpy())
            targets.extend(age.numpy())

    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print(f"MAE: {mae:.2f} years")
    print(f"R²:  {r2:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"scatter_{args.mode}.png"
    plot_predictions(targets, predictions, mae, r2, args.mode, out_path)
    print(f"Scatter plot saved to: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)