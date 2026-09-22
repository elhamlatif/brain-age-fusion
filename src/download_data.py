"""
Downloads the LEMON dataset (ds000221) from OpenNeuro -- just T1 + resting
EEG, not the full thing, because the full dataset is like 80GB and I do
not have that kind of space on this machine.

    pip install openneuro-py
    python src/download_data.py --n-subjects 20
"""

import argparse
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_ID = "ds000221"


def get_subject_ids(n_subjects):
    import openneuro

    # participants.tsv is like 10KB so just grab it separately first,
    # then we know which subjects actually exist before pulling anything big
    openneuro.download(
        dataset=DATASET_ID,
        target_dir=RAW_DIR,
        include=["participants.tsv"],
    )

    with open(RAW_DIR / "participants.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    ids = [row["participant_id"] for row in rows]
    return ids[:n_subjects]


def download(n_subjects):
    try:
        import openneuro
    except ImportError:
        print("pip install openneuro-py first")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subject_ids = get_subject_ids(n_subjects)

    print(f"got {len(subject_ids)} subject ids, downloading T1 + EEG for those...")

    include = ["participants.tsv"]
    for sid in subject_ids:
        # not sure every subject has ses-01 vs ses-02, LEMON is a bit
        # inconsistent about this, so just glob both and let missing
        # ones be missing
        include.append(f"{sid}/ses-*/anat/*T1w.nii.gz")
        include.append(f"{sid}/ses-*/eeg/*eyesclosed*")

    openneuro.download(
        dataset=DATASET_ID,
        target_dir=RAW_DIR,
        include=include,
    )
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=20)
    download(parser.parse_args().n_subjects)