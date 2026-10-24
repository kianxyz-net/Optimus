"""
The cross-species arm.

Read this before you put a number on a slide.

WHAT THIS DOES NOT DO: it does not train on mouse or monkey brains and predict
human cognitive load. Those weights do not transfer. Different cortical
geometry, different electrode physics, no channel correspondence, and no
labelled "cognitive load" condition in any animal EEG set I could find. Anyone
claiming otherwise is reporting noise.

WHAT THIS DOES DO, and what is defensible to present:

  (a) SHARED EXTRACTOR. The same feature code runs on mouse, macaque and human
      recordings. If frontal-theta-up / alpha-down is a real signature of
      neural load rather than an artefact of one montage, the extractor should
      detect a task-versus-rest spectral shift in any species. Showing the same
      pipeline move in the same direction in two species is a genuine finding
      and it is cheap to produce.

  (b) STATE-CHANGE DETECTION AS A SPECIES-FREE TASK. Drop the "cognitive load"
      label and ask a narrower question: can this feature set separate
      stimulated from unstimulated windows? That question is well posed in
      animal data (the Hwang mouse set has sound and optogenetic stimulation
      blocks) and it validates the extractor without any label-transfer claim.

  (c) CORAL ALIGNMENT, for the case where you do want to pool domains. CORAL
      whitens the source covariance and recolours it to the target. It is the
      standard cheap domain adaptation for EEG and it is honest about what it
      is: a statistical alignment, not a translation between brains. Use it
      human-to-human (STEW to EEGMAT, or dataset to your live headset) where
      it actually helps. Using it mouse-to-human will produce numbers; those
      numbers will not mean what you want them to mean.

The honest headline for a hackathon: "we trained on human data, and we show
the same extractor responds in mouse data." That is a real translational
result. It is also the one you can defend when a judge pushes.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .features import extract_recording, BaselineNormalizer


def coral(source: np.ndarray, target: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """CORAL: align source feature covariance to target.

    Returns the transformed source. Use when training on one domain (a public
    dataset) and testing on another (your live headset), and report the
    unaligned number alongside so the gain is visible.
    """
    Xs = np.asarray(source, dtype=float)
    Xt = np.asarray(target, dtype=float)

    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + reg * np.eye(Xt.shape[1])

    Ws = _inv_sqrtm(Cs)
    Wt = _sqrtm(Ct)
    return (Xs - Xs.mean(0)) @ Ws @ Wt + Xt.mean(0)


def _sqrtm(C):
    vals, vecs = linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _inv_sqrtm(C):
    vals, vecs = linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def spectral_shift(
    baseline: np.ndarray,
    active: np.ndarray,
    sfreq: float,
    species: str,
    ch_names: list[str] | None = None,
    win_sec: float = 2.0,
) -> dict:
    """Quantify how far an active block sits from a baseline block.

    Runs in any species. Returns per-feature effect sizes (Cohen's d) plus a
    single scalar distance, so you can put mouse and human on the same axis
    without claiming the underlying states are the same.
    """
    Xb, names = extract_recording(
        baseline, sfreq, ch_names, species=species, win_sec=win_sec
    )
    Xa, _ = extract_recording(
        active, sfreq, ch_names, species=species, win_sec=win_sec
    )

    norm = BaselineNormalizer().fit(Xb)
    Za = norm.transform(Xa)
    Zb = norm.transform(Xb)

    pooled_sd = np.sqrt((Za.var(0) + Zb.var(0)) / 2.0)
    pooled_sd[pooled_sd < 1e-9] = 1.0
    d = (Za.mean(0) - Zb.mean(0)) / pooled_sd

    return {
        "species": species,
        "feature_names": names,
        "cohens_d": dict(sorted(zip(names, d), key=lambda kv: -abs(kv[1]))),
        "mean_abs_z": float(np.mean(np.abs(Za.mean(0)))),
        "n_baseline_windows": int(Xb.shape[0]),
        "n_active_windows": int(Xa.shape[0]),
    }


def compare_species(results: list[dict]) -> dict:
    """Line up spectral_shift results from several species.

    Reports which features moved in the SAME DIRECTION in every species. That
    agreement set is the actual cross-species claim: not "the model transfers"
    but "these spectral features respond to task demand in more than one
    brain".
    """
    if not results:
        return {}

    common = set(results[0]["cohens_d"])
    for r in results[1:]:
        common &= set(r["cohens_d"])

    agreement = {}
    for feat in sorted(common):
        ds = [r["cohens_d"][feat] for r in results]
        signs = {np.sign(v) for v in ds if abs(v) > 0.2}
        agreement[feat] = {
            "per_species": {r["species"]: round(r["cohens_d"][feat], 3) for r in results},
            "consistent": len(signs) == 1 and len(ds) > 1,
        }

    consistent = [k for k, v in agreement.items() if v["consistent"]]
    return {
        "species": [r["species"] for r in results],
        "features": agreement,
        "consistent_features": consistent,
        "n_consistent": len(consistent),
    }
