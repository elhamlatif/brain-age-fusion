"""
Train the multimodal brain-age model.

Loss is L1 (i.e. MAE in years) — the same loss almost every brain-age paper
has used since Cole et al.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import MultimodalBrainDataset
from model import FusionModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Keep these in sync with evaluate.py
VAL_FRACTION = 0.2
SEED = 42


def build_loaders(batch_size):
    ds = MultimodalBrainDataset()
    n_eeg = ds[0][1].shape[0]

    all_idx = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=VAL_FRACTION, random_state=SEED
    )

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, train_ds, val_ds, n_eeg


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for mri, eeg, age in tqdm(loader, desc="  train", leave=False):
        mri = mri.to(device, non_blocking=True)
        eeg = eeg.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(mri, eeg)
        loss = criterion(pred, age)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mri.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_abs_error = 0.0

    for mri, eeg, age in loader:
        mri = mri.to(device, non_blocking=True)
        eeg = eeg.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)

        pred = model(mri, eeg)
        total_abs_error += torch.abs(pred - age).sum().item()

    return total_abs_error / len(loader.dataset)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.mode}] device={device}  epochs={args.epochs}  lr={args.lr}")

    train_loader, val_loader, train_ds, val_ds, n_eeg = build_loaders(args.batch_size)
    print(f"  split: train={len(train_ds)}  val={len(val_ds)}  n_eeg={n_eeg}")

    model = FusionModel(n_eeg_features=n_eeg, mode=args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / f"best_model_{args.mode}.pt"

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mae = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:3d}/{args.epochs}: "
            f"train={train_mae:6.2f}  val={val_mae:6.2f}"
        )

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"           * new best ({best_val:.2f}) → {ckpt_path.name}")

    print(f"\nfinished {args.mode}: best val MAE = {best_val:.2f} years")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multimodal brain-age model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--mode",
        choices=["fusion", "mri_only", "eeg_only"],
        default="fusion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())