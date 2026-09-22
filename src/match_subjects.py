"""
Build a list of subjects that have both processed MRI volumes and EEG
feature vectors, and merge their ages from participants.tsv.

Output: data/processed/subjects.csv with columns [subject_id, age]
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

OUTPUT_CSV = PROCESSED_DIR / "subjects.csv"
PARTICIPANTS_TSV = RAW_DIR / "participants.tsv"
PARTICIPANT_ID_COLUMN = "participant_id"


def list_processed_subjects(folder):
    if not folder.is_dir():
        return set()
    return {path.stem for path in folder.glob("*.npy")}


def find_common_subjects():
    mri_subjects = list_processed_subjects(PROCESSED_DIR / "mri")
    eeg_subjects = list_processed_subjects(PROCESSED_DIR / "eeg")
    common_subjects = sorted(mri_subjects & eeg_subjects)

    print(f"MRI: {len(mri_subjects)}, EEG: {len(eeg_subjects)}, both: {len(common_subjects)}")

    return common_subjects


def load_ages(subjects):
    if not PARTICIPANTS_TSV.exists():
        print("participants.tsv not found, ages will be missing for all subjects.")
        return {sid: None for sid in subjects}

    participants = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    participants = participants.set_index(PARTICIPANT_ID_COLUMN)

    ages = {}
    for sid in subjects:
        if sid in participants.index:
            ages[sid] = participants.loc[sid].get("age", None)
        else:
            ages[sid] = None
    return ages


def save_subjects_csv(subjects, ages):
    rows = [{"subject_id": sid, "age": ages.get(sid)} for sid in subjects]
    df = pd.DataFrame(rows)

    n_missing = df["age"].isna().sum()
    if n_missing:
        print(f"Dropping {n_missing} subjects with no age in participants.tsv.")
    df = df.dropna()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return df


def run():
    common_subjects = find_common_subjects()

    if not common_subjects:
        print("No subjects with both modalities found. Skipping CSV creation.")
        return

    ages = load_ages(common_subjects)
    df = save_subjects_csv(common_subjects, ages)

    print(f"Saved {len(df)} subjects to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()