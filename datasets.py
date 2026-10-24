"""
Loaders for the public cognitive-load datasets.

None of these are bundled -- download them yourself (see README) and point the
loaders at the folder. Each loader returns a list of Recording objects so the
rest of the pipeline never has to know which dataset it came from.

STEW is the primary training set because its hardware matches what is on the
hackathon device list: a 14-channel Emotiv EPOC. A model trained on EPOC data
and run on an EPOC headset skips the hardest part of the transfer problem.

EEGMAT is the second set: different hardware, different country, different
task. It exists so you can report cross-DATASET accuracy, not just
cross-subject. That number is the honest one.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Recording:
    """One continuous EEG block with a known load condition."""

    data: np.ndarray                 # (n_channels, n_samples)
    sfreq: float
    ch_names: list[str]
    subject: str
    condition: str                   # "rest" | "task"
    species: str = "human"
    label: int | None = None         # 0 = low load, 1 = high load
    rating: float | None = None      # subjective rating where available
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# STEW -- Simultaneous Task EEG Workload
# 48 subjects, Emotiv EPOC, 14 channels, 128 Hz, 2.5 min per condition.
# Files: subXX_lo.txt (rest) and subXX_hi.txt (SIMKAP multitasking).
# ratings.txt: "subject, rest_rating, task_rating" on a 1-9 scale.
# https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset
# --------------------------------------------------------------------------

STEW_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
STEW_SFREQ = 128.0


def load_stew(root: str, subjects: list[str] | None = None) -> list[Recording]:
    """Load the STEW dataset from a directory of .txt files."""
    ratings = _load_stew_ratings(os.path.join(root, "ratings.txt"))

    recs: list[Recording] = []
    for path in sorted(glob.glob(os.path.join(root, "sub*_*.txt"))):
        base = os.path.basename(path)
        if base.startswith("ratings"):
            continue
        stem = base.rsplit(".", 1)[0]
        try:
            sub_part, cond_part = stem.split("_")
        except ValueError:
            continue

        sub = sub_part.replace("sub", "").lstrip("0") or "0"
        if subjects is not None and sub not in subjects:
            continue

        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] != len(STEW_CHANNELS):
            # a few files in the wild have stray whitespace columns
            arr = arr[:, : len(STEW_CHANNELS)]

        condition = "rest" if cond_part == "lo" else "task"
        rating = ratings.get(sub, {}).get(condition)

        recs.append(
            Recording(
                data=arr.T.astype(float),
                sfreq=STEW_SFREQ,
                ch_names=list(STEW_CHANNELS),
                subject=f"stew-{sub}",
                condition=condition,
                species="human",
                label=0 if condition == "rest" else 1,
                rating=rating,
                meta={"dataset": "STEW"},
            )
        )
    return recs


def _load_stew_ratings(path: str) -> dict:
    out: dict = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) < 3:
                continue
            try:
                sub = str(int(parts[0]))
                out[sub] = {"rest": float(parts[1]), "task": float(parts[2])}
            except ValueError:
                continue
    return out


# --------------------------------------------------------------------------
# EEGMAT -- EEG During Mental Arithmetic Tasks (PhysioNet)
# 36 subjects, Neurocom 23-channel, 10/20 montage, EDF.
# SubjectXX_1.edf = 180 s rest, SubjectXX_2.edf = 60 s serial subtraction.
# https://physionet.org/content/eegmat/
# --------------------------------------------------------------------------


def load_eegmat(root: str, subjects: list[str] | None = None) -> list[Recording]:
    """Load EEGMAT from a directory of .edf files. Requires mne."""
    import mne

    recs: list[Recording] = []
    for path in sorted(glob.glob(os.path.join(root, "*.edf"))):
        stem = os.path.basename(path).rsplit(".", 1)[0]
        if "_" not in stem:
            continue
        sub_part, which = stem.rsplit("_", 1)
        sub = sub_part.lower().replace("subject", "").lstrip("0") or "0"
        if subjects is not None and sub not in subjects:
            continue

        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        condition = "rest" if which == "1" else "task"

        # EDF labels look like "EEG Fp1-A1"; reduce to the electrode name
        names = [_clean_edf_label(n) for n in raw.ch_names]

        recs.append(
            Recording(
                data=raw.get_data().astype(float),
                sfreq=float(raw.info["sfreq"]),
                ch_names=names,
                subject=f"eegmat-{sub}",
                condition=condition,
                species="human",
                label=0 if condition == "rest" else 1,
                meta={"dataset": "EEGMAT"},
            )
        )
    return recs


def _clean_edf_label(label: str) -> str:
    s = label.replace("EEG", "").strip()
    for sep in ("-",):
        if sep in s:
            s = s.split(sep)[0]
    return s.strip().upper()


# --------------------------------------------------------------------------
# Animal EEG -- generic loader
#
# Used for the cross-species arm. Point it at an EEGLAB .set (the Hwang mouse
# hdEEG set) or any file mne can read. There is no load label here: the animal
# data validates that the feature extractor responds to stimulation, it does
# NOT contribute training labels for the human decoder. See README.
# --------------------------------------------------------------------------


def load_animal(path: str, species: str = "mouse", subject: str | None = None):
    """Load one animal recording via mne. Returns a single Recording."""
    import mne

    ext = os.path.splitext(path)[1].lower()
    if ext == ".set":
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        except Exception:
            epochs = mne.io.read_epochs_eeglab(path, verbose="ERROR")
            data = np.concatenate(list(epochs.get_data()), axis=-1)
            return Recording(
                data=data.astype(float),
                sfreq=float(epochs.info["sfreq"]),
                ch_names=list(epochs.ch_names),
                subject=subject or os.path.basename(path),
                condition="stim",
                species=species,
                meta={"source": path, "epoched": True},
            )
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    else:
        raw = mne.io.read_raw(path, preload=True, verbose="ERROR")

    return Recording(
        data=raw.get_data().astype(float),
        sfreq=float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        subject=subject or os.path.basename(path),
        condition="stim",
        species=species,
        meta={"source": path},
    )
