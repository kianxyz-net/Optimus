"""
Feature extraction for cognitive-load decoding.

Design constraint that drives everything here: the feature vector must have the
same meaning whether it came from a 14-channel Emotiv at 128 Hz, a 23-channel
clinical cap at 500 Hz, or a mouse epidural electrode at 1 kHz.

Three mechanisms get us there:

1. RELATIVE band power. Every band is divided by total in-band power, so the
   absolute microvolt scale cancels. This kills amplifier gain, electrode
   impedance, skull thickness, and species amplitude differences in one step.

2. REGION AVERAGING (human/macaque). Channels collapse into frontal / central /
   posterior, so montage size stops mattering.

3. PER-INDIVIDUAL BASELINE z-scoring. Each subject's (or animal's) features are
   z-scored against their own rest recording. This removes the individual
   differences that otherwise dominate EEG, and it is what makes a model trained
   on one population usable on a new person without retraining.

After step 3 a feature vector is a statement about "how far is this window from
this individual's own resting spectrum", which is a quantity that exists in any
brain with a measurable EEG spectrum. That is the honest basis for cross-species
comparison -- not shared weights.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .bands import bands_for, region_of

EPS = 1e-12


def _welch_psd(x: np.ndarray, sfreq: float, nperseg_sec: float = 1.0):
    """Welch PSD over the last axis. x: (..., n_samples)."""
    nperseg = int(round(nperseg_sec * sfreq))
    nperseg = max(32, min(nperseg, x.shape[-1]))
    freqs, psd = signal.welch(
        x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1
    )
    return freqs, psd


def band_powers(
    x: np.ndarray,
    sfreq: float,
    species: str = "human",
    relative: bool = True,
) -> dict:
    """Band power per channel.

    x        : (n_channels, n_samples)
    returns  : {band_name: (n_channels,) array}
    """
    bands = bands_for(species)
    freqs, psd = _welch_psd(x, sfreq)

    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            # band sits above Nyquist for this sampling rate
            out[name] = np.zeros(psd.shape[0])
            continue
        out[name] = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)

    if relative:
        total = np.sum(list(out.values()), axis=0) + EPS
        out = {k: v / total for k, v in out.items()}
    return out


def _region_reduce(per_channel: dict, ch_names: list[str]) -> dict:
    """Average band powers within frontal/central/posterior regions.

    Falls back to a single 'all' region when no channel label is recognised,
    which is the normal case for animal montages.
    """
    assignments = [region_of(c) for c in ch_names]
    known = [a for a in assignments if a is not None]

    if not known:
        return {f"all_{band}": float(np.mean(v)) for band, v in per_channel.items()}

    feats = {}
    for region in ("frontal", "central", "posterior"):
        idx = [i for i, a in enumerate(assignments) if a == region]
        for band, vals in per_channel.items():
            if idx:
                feats[f"{region}_{band}"] = float(np.mean(vals[idx]))
            else:
                feats[f"{region}_{band}"] = float(np.mean(vals))
    return feats


def _load_ratios(feats: dict) -> dict:
    """Classic workload indices, built from whatever regions exist.

    Frontal theta rises and posterior alpha falls as task load increases; the
    theta/alpha ratio is the single most replicated EEG workload index. These
    are added explicitly rather than left for the classifier to discover,
    because with a few hundred training windows an explicit feature beats a
    learned interaction.
    """
    extra = {}

    def g(key):
        return feats.get(key, np.nan)

    f_theta = g("frontal_theta")
    p_alpha = g("posterior_alpha")
    if np.isfinite(f_theta) and np.isfinite(p_alpha):
        extra["theta_alpha_ratio"] = float(f_theta / (p_alpha + EPS))
    elif "all_theta" in feats and "all_alpha" in feats:
        extra["theta_alpha_ratio"] = float(
            feats["all_theta"] / (feats["all_alpha"] + EPS)
        )

    theta_keys = [k for k in feats if k.endswith("_theta")]
    alpha_keys = [k for k in feats if k.endswith("_alpha")]
    beta_keys = [k for k in feats if k.endswith("_beta")]
    if theta_keys and beta_keys and alpha_keys:
        t = np.mean([feats[k] for k in theta_keys])
        a = np.mean([feats[k] for k in alpha_keys])
        b = np.mean([feats[k] for k in beta_keys])
        # engagement index; higher = more engaged / loaded
        extra["engagement"] = float(b / (a + t + EPS))

    return extra


def _spectral_entropy(x: np.ndarray, sfreq: float) -> float:
    """Normalised spectral entropy, averaged over channels.

    Rises when the spectrum flattens, which it tends to do under load as the
    dominant resting rhythm breaks up. Species-agnostic by construction.
    """
    freqs, psd = _welch_psd(x, sfreq)
    band = (freqs >= 1.0) & (freqs <= 45.0)
    p = psd[..., band]
    p = p / (p.sum(axis=-1, keepdims=True) + EPS)
    h = -np.sum(p * np.log(p + EPS), axis=-1)
    return float(np.mean(h / np.log(p.shape[-1])))


def extract_window(
    x: np.ndarray,
    sfreq: float,
    ch_names: list[str] | None = None,
    species: str = "human",
) -> dict:
    """Full feature dict for one window of EEG.

    x : (n_channels, n_samples)
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(x.shape[0])]

    # detrend removes slow drift that otherwise lands in delta
    x = signal.detrend(x, axis=-1, type="linear")

    per_ch = band_powers(x, sfreq, species=species, relative=True)
    feats = _region_reduce(per_ch, ch_names)
    feats.update(_load_ratios(feats))
    feats["spectral_entropy"] = _spectral_entropy(x, sfreq)
    return feats


def windows(
    x: np.ndarray, sfreq: float, win_sec: float = 2.0, step_sec: float = 1.0
):
    """Yield (start_sample, window) sliding over a continuous recording."""
    n = int(round(win_sec * sfreq))
    step = int(round(step_sec * sfreq))
    for start in range(0, x.shape[-1] - n + 1, step):
        yield start, x[..., start : start + n]


def extract_recording(
    x: np.ndarray,
    sfreq: float,
    ch_names: list[str] | None = None,
    species: str = "human",
    win_sec: float = 2.0,
    step_sec: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for a whole recording.

    returns (n_windows, n_features) and the feature names.
    """
    rows, names = [], None
    for _, w in windows(x, sfreq, win_sec, step_sec):
        f = extract_window(w, sfreq, ch_names, species)
        if names is None:
            names = sorted(f)
        rows.append([f[k] for k in names])
    if not rows:
        raise ValueError("recording shorter than one window")
    return np.asarray(rows), names


class BaselineNormalizer:
    """Per-individual z-scoring against that individual's own rest recording.

    This is the piece that makes the model portable across people, headsets and
    species. Fit it on a rest/baseline block, then transform task windows.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, baseline_features: np.ndarray) -> "BaselineNormalizer":
        b = np.asarray(baseline_features, dtype=float)
        self.mean_ = np.nanmean(b, axis=0)
        self.std_ = np.nanstd(b, axis=0)
        self.std_[self.std_ < EPS] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("call fit() on a baseline block first")
        z = (np.asarray(features, dtype=float) - self.mean_) / self.std_
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, baseline_features, features=None) -> np.ndarray:
        self.fit(baseline_features)
        return self.transform(baseline_features if features is None else features)
