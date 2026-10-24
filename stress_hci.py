"""
Loader for: Neurophysiological Dataset of Stress Resilience During
Human-Computer Interaction (PhysioNet, doi 10.13026/x3vc-p627).

FILE LAYOUT. Each participant CSV has 74 ROWS and one column per sample --
rows are signals, columns are time. That is already the (n_channels,
n_samples) orientation the feature code wants, so no transpose.

    row  1       timestamp
    rows 2-17    fNIRS optical density        (16)
    rows 18-33   fNIRS concentration HbO/HbR  (16)
    rows 34-65   EEG, 32 channels, 10-20      (32)
    rows 66-73   eye tracking                 (8)
    row  74      event markers

Those are 1-indexed as written in the dataset docs; the constants below are
0-indexed.

CHANNEL ORDER WARNING. The dataset documentation lists the 32 EEG channels in
two different orders -- once as prose and once with explicit electrode
numbers, and they disagree at positions 4-7 and 20-22 (F7/F8/F3/Fz/F4 and
CP3/CP4/CP1/CP2 are permuted between the two). We use the NUMBERED order,
since explicit indices are the stronger evidence. This mostly does not matter
here: every disagreeing pair sits in the same scalp region, and the feature
extractor averages within region before doing anything else. That is a real
argument for region averaging over per-channel features on a dataset you did
not record yourself.

Sampling: 250 Hz for EEG, fNIRS and (for p01-p24) eye tracking.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .datasets import Recording
from .protocol import (
    PHASES,
    PHASE_LEVEL,
    EXPECTED_MARKERS,
    MARKER_PHASES,
    expected_boundaries_sec,
)

SFREQ = 250.0

# 0-indexed row slices
ROW_TIMESTAMP = 0
ROWS_FNIRS_OD = slice(1, 17)
ROWS_FNIRS_CONC = slice(17, 33)
ROWS_EEG = slice(33, 65)
ROWS_EYE = slice(65, 73)
ROW_MARKERS = 73

N_ROWS = 74

# numbered order from the "EEG Channel Locations" section
EEG_CHANNELS = [
    "AF3", "AF4", "F7", "F8", "F3", "Fz", "F4", "FT7",
    "FT8", "FC5", "FC3", "FC4", "FC6", "C5", "C1", "Cz",
    "C2", "C6", "CP3", "CP4", "CP1", "CP2", "P7", "P3",
    "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "Oz",
]
assert len(EEG_CHANNELS) == 32

# prose order, kept so you can switch and see whether it changes anything.
# If your results move when you swap these, you are over-relying on individual
# channels and should lean harder on the region features.
EEG_CHANNELS_PROSE_ORDER = [
    "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FT7",
    "FT8", "FC5", "FC3", "FC4", "FC6", "C5", "C1", "Cz",
    "C2", "C6", "CP3", "CP1", "CP2", "CP4", "P7", "P3",
    "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "Oz",
]


# --------------------------------------------------------------------------
# raw file reading
# --------------------------------------------------------------------------


def read_participant_csv(path: str) -> dict:
    """Read one participant CSV into its component modalities."""
    arr = np.loadtxt(path, delimiter=",", dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2-D array, got shape {arr.shape}")

    # Guard against a transposed file. Recordings are ~31 min at 250 Hz, so
    # the time axis is ~465k samples and the signal axis is 74. If the short
    # axis is on the wrong side, flip it rather than failing.
    if arr.shape[0] != N_ROWS and arr.shape[1] == N_ROWS:
        arr = arr.T
    if arr.shape[0] != N_ROWS:
        raise ValueError(
            f"{path}: expected {N_ROWS} signal rows, got shape {arr.shape}. "
            "File layout may differ for this participant -- inspect it."
        )

    return {
        "timestamp": arr[ROW_TIMESTAMP],
        "fnirs_od": arr[ROWS_FNIRS_OD],
        "fnirs_conc": arr[ROWS_FNIRS_CONC],
        "eeg": arr[ROWS_EEG],
        "eye": arr[ROWS_EYE],
        "markers": arr[ROW_MARKERS],
        "n_samples": arr.shape[1],
    }


def marker_samples(markers: np.ndarray, min_gap_sec: float = 60.0) -> list[int]:
    """Sample indices where a marker fires.

    Markers are a mostly-zero row with nonzero pulses. A pulse can span
    several samples, so adjacent detections within min_gap_sec collapse into
    one event.
    """
    idx = np.flatnonzero(np.nan_to_num(markers) != 0)
    if idx.size == 0:
        return []

    min_gap = int(min_gap_sec * SFREQ)
    events = [int(idx[0])]
    for i in idx[1:]:
        if i - events[-1] >= min_gap:
            events.append(int(i))
    return events


# --------------------------------------------------------------------------
# phase segmentation
# --------------------------------------------------------------------------


def segment_phases(
    n_samples: int,
    markers: np.ndarray,
    strict: bool = True,
) -> tuple[list[tuple[str, int, int]], str]:
    """Split a recording into (phase_name, start_sample, end_sample).

    Returns the segments and a string describing how they were derived, so
    you can report how many subjects needed the fallback instead of quietly
    mixing two labelling methods.

    Marker semantics: the first marker starts working_baseline, and each
    subsequent marker starts the next phase. Everything before marker 1 is
    resting_baseline; everything after the last marker is recovery_2.
    """
    events = marker_samples(markers)

    if len(events) != EXPECTED_MARKERS:
        if strict:
            return [], (
                f"marker count {len(events)} != {EXPECTED_MARKERS}; skipped"
            )
        segs = []
        for name, t0, t1 in expected_boundaries_sec():
            s0, s1 = int(t0 * SFREQ), int(t1 * SFREQ)
            if s0 >= n_samples:
                break
            segs.append((name, s0, min(s1, n_samples)))
        return segs, (
            f"marker count {len(events)} != {EXPECTED_MARKERS}; "
            "used nominal protocol timing"
        )

    segs = [("resting_baseline", 0, events[0])]
    for i, name in enumerate(MARKER_PHASES):
        start = events[i]
        end = events[i + 1] if i + 1 < len(events) else n_samples
        segs.append((name, start, end))

    # drop anything implausibly short -- a phase under 60 s means the markers
    # fired wrong even though the count looked right
    good = [(n, a, b) for n, a, b in segs if (b - a) >= int(60 * SFREQ)]
    if len(good) != len(segs):
        dropped = [n for n, a, b in segs if (b - a) < int(60 * SFREQ)]
        return good, f"markers ok but dropped short phases: {dropped}"

    return segs, "markers"


# --------------------------------------------------------------------------
# public loader
# --------------------------------------------------------------------------


def load_stress_hci(
    root: str,
    subjects: list[str] | None = None,
    modality: str = "eeg",
    strict_markers: bool = True,
    verbose: bool = True,
) -> list[Recording]:
    """Load the stress-resilience dataset as per-phase Recordings.

    root           : directory containing the EEGfNIRSeye_* folders, or any
                     directory of participant CSVs
    modality       : "eeg" | "fnirs" -- which signal to put in Recording.data
    strict_markers : skip subjects whose markers don't parse, rather than
                     falling back to nominal timing. Default True because a
                     mislabelled subject is worse than a missing one.

    Each returned Recording is ONE PHASE of one participant, with
    Recording.label set to the ordinal load level (0/1/2) and
    Recording.condition set to "rest" for the resting baseline (so the
    baseline normaliser can find it) or the phase name otherwise.
    """
    paths = _find_participant_files(root)
    if not paths:
        raise FileNotFoundError(
            f"no participant CSVs under {root!r}. Expected files like "
            "p01.csv inside EEGfNIRSeye_p1-p5/ etc."
        )

    recs: list[Recording] = []
    stats = {"loaded": 0, "skipped": 0, "fallback": 0}

    for path in paths:
        pid = _participant_id(path)
        if subjects is not None and pid not in subjects:
            continue
        if pid in {"p14", "p23"}:
            # excluded by the dataset authors for missing values
            continue

        try:
            parts = read_participant_csv(path)
        except Exception as exc:
            if verbose:
                print(f"  {pid}: unreadable ({exc})")
            stats["skipped"] += 1
            continue

        segs, how = segment_phases(
            parts["n_samples"], parts["markers"], strict=strict_markers
        )
        if not segs:
            if verbose:
                print(f"  {pid}: {how}")
            stats["skipped"] += 1
            continue
        if how != "markers":
            stats["fallback"] += 1
            if verbose:
                print(f"  {pid}: {how}")

        signal = parts["eeg"] if modality == "eeg" else parts["fnirs_conc"]
        ch_names = (
            list(EEG_CHANNELS)
            if modality == "eeg"
            else [f"fnirs{i}" for i in range(signal.shape[0])]
        )

        for name, s0, s1 in segs:
            recs.append(
                Recording(
                    data=signal[:, s0:s1],
                    sfreq=SFREQ,
                    ch_names=ch_names,
                    subject=f"shci-{pid}",
                    condition="rest" if name == "resting_baseline" else name,
                    species="human",
                    label=PHASE_LEVEL[name],
                    meta={
                        "dataset": "stress-hci",
                        "phase": name,
                        "modality": modality,
                        "segmentation": how,
                        "fnirs": parts["fnirs_conc"][:, s0:s1]
                        if modality == "eeg"
                        else None,
                    },
                )
            )
        stats["loaded"] += 1

    if verbose:
        print(
            f"  loaded {stats['loaded']} participants "
            f"({stats['skipped']} skipped, {stats['fallback']} on nominal timing)"
        )
    return recs


def _find_participant_files(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "EEGfNIRSeye_*", "*.csv"),
        os.path.join(root, "*", "EEGfNIRSeye_*", "*.csv"),
        os.path.join(root, "p*.csv"),
        os.path.join(root, "*", "p*.csv"),
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(glob.glob(pat))

    # drop the sidecar csvs that are not participant recordings
    skip = ("resman", "eye", "demographic", "tags", "hr", "eda", "bvp", "ibi",
            "acc", "temp")
    out = []
    for p in sorted(set(found)):
        base = os.path.basename(p).lower()
        if any(s in base for s in skip):
            continue
        out.append(p)
    return out


def _participant_id(path: str) -> str:
    base = os.path.basename(path).rsplit(".", 1)[0].lower()
    for tok in base.replace("-", "_").split("_"):
        if tok.startswith("p") and tok[1:].isdigit():
            return f"p{int(tok[1:]):02d}"
    return base
