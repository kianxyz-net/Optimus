"""
Ordinal load decoder.

The three levels (rest < task < high load) are ORDERED, and a plain 3-class
classifier throws that away. It treats confusing rest with high-load as the
same size of mistake as confusing task with high-load, which is wrong both
statistically and for the demo: a meter that jumps from 0 to 2 looks broken in
a way that drifting 1 to 2 does not.

Method: cumulative binary decomposition (Frank & Hall). For K levels, fit K-1
calibrated binary models:

    model_0 : P(y > 0)   "is anything happening at all"
    model_1 : P(y > 1)   "is this the hard condition"

then

    P(y = 0) = 1 - P(y > 0)
    P(y = 1) = P(y > 0) - P(y > 1)
    P(y = 2) = P(y > 1)

Two reasons this beats multinomial here:

1. Each sub-model is binary, so sklearn's probability calibration applies
   directly, and the per-level probabilities stay honest.
2. The expected level, sum_k k * P(y = k), is a continuous scalar in [0, 2].
   That is the number you put on the live meter. It moves smoothly, which a
   hard argmax does not.

Monotonicity is enforced after the fact: the cumulative probabilities must be
non-increasing, and independently-fitted sub-models occasionally violate that
near the boundary. We clip rather than pretend it never happens.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold

from .model import LoadDecoder
from .protocol import N_LEVELS, LEVEL_NAMES


class OrdinalLoadDecoder:
    """Ordinal decoder over load levels 0..K-1."""

    def __init__(self, n_levels: int = N_LEVELS, kind: str = "logreg",
                 calibration: str = "sigmoid", seed: int = 0):
        self.n_levels = n_levels
        self.kind = kind
        self.calibration = calibration
        self.seed = seed
        self.models_: list[LoadDecoder] = []
        self.feature_names_: list[str] | None = None

    def fit(self, X, y, groups=None, feature_names=None) -> "OrdinalLoadDecoder":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        self.models_ = []
        for k in range(self.n_levels - 1):
            yk = (y > k).astype(int)
            if len(np.unique(yk)) < 2:
                raise ValueError(
                    f"threshold {k} has only one class present; "
                    "you are missing a load level in the training data"
                )
            dec = LoadDecoder(self.kind, self.calibration, self.seed)
            dec.fit(X, yk, groups=groups, feature_names=feature_names)
            self.models_.append(dec)
        return self

    def predict_cumulative(self, X) -> np.ndarray:
        """(n_samples, n_levels-1) array of P(y > k), forced non-increasing."""
        cum = np.column_stack([m.predict_load(X) for m in self.models_])
        # P(y>0) >= P(y>1) >= ... must hold; enforce by running minimum
        return np.minimum.accumulate(cum, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """(n_samples, n_levels) per-level probabilities."""
        cum = self.predict_cumulative(X)
        n = cum.shape[0]
        p = np.zeros((n, self.n_levels))
        p[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, self.n_levels - 1):
            p[:, k] = cum[:, k - 1] - cum[:, k]
        p[:, -1] = cum[:, -1]
        return np.clip(p, 0.0, 1.0)

    def predict_level(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def expected_level(self, X) -> np.ndarray:
        """Continuous load estimate in [0, n_levels-1]. This is the meter."""
        p = self.predict_proba(X)
        return p @ np.arange(self.n_levels)

    def confidence(self, X) -> np.ndarray:
        """Probability mass on the predicted level."""
        p = self.predict_proba(X)
        return p.max(axis=1)

    def coefficients(self, threshold: int = 1) -> dict | None:
        """Weights of one sub-model. threshold=1 is the rest-vs-high boundary."""
        if threshold >= len(self.models_):
            return None
        return self.models_[threshold].coefficients()


def decide_level(p_levels: np.ndarray, min_conf: float = 0.5) -> str:
    """Verdict with abstention, for a single window's probability vector."""
    k = int(np.argmax(p_levels))
    if p_levels[k] < min_conf:
        return "uncertain"
    return LEVEL_NAMES.get(k, str(k))


def evaluate_ordinal(X, y, groups, kind: str = "logreg", n_splits: int = 5,
                     seed: int = 0) -> dict:
    """Subject-wise CV for the ordinal decoder.

    Reports MAE in load levels alongside accuracy, because for an ordered
    target MAE is the metric that matches what the user sees. An accuracy of
    0.6 with MAE 0.42 is a usable meter; accuracy 0.6 with MAE 0.9 is not,
    and plain accuracy cannot tell them apart.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("need at least 2 subjects")

    oof_level = np.full(len(y), -1, dtype=int)
    oof_expected = np.full(len(y), np.nan)
    oof_conf = np.full(len(y), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        dec = OrdinalLoadDecoder(kind=kind, seed=seed)
        dec.fit(X[tr], y[tr], groups=groups[tr])
        oof_level[te] = dec.predict_level(X[te])
        oof_expected[te] = dec.expected_level(X[te])
        oof_conf[te] = dec.confidence(X[te])

    # chance MAE: always guess the most common level
    majority = int(np.bincount(y).argmax())
    chance_mae = float(mean_absolute_error(y, np.full_like(y, majority)))

    confident = oof_conf >= 0.5
    per_level = {}
    for k in range(N_LEVELS):
        m = y == k
        if m.any():
            per_level[LEVEL_NAMES[k]] = {
                "n": int(m.sum()),
                "recall": float(np.mean(oof_level[m] == k)),
                "mean_estimate": float(np.nanmean(oof_expected[m])),
            }

    return {
        "n_windows": int(len(y)),
        "n_subjects": int(n_groups),
        "accuracy": float(accuracy_score(y, oof_level)),
        "majority_baseline": float(np.mean(y == majority)),
        "mae": float(mean_absolute_error(y, oof_level)),
        "chance_mae": chance_mae,
        "adjacent_accuracy": float(np.mean(np.abs(oof_level - y) <= 1)),
        "confusion_matrix": confusion_matrix(y, oof_level,
                                             labels=list(range(N_LEVELS))).tolist(),
        "per_level": per_level,
        "coverage_at_conf_50": float(np.mean(confident)),
        "accuracy_when_confident": (
            float(accuracy_score(y[confident], oof_level[confident]))
            if confident.any() else float("nan")
        ),
        "oof_expected": oof_expected,
        "oof_level": oof_level,
    }


def recovery_check(expected_by_phase: dict) -> dict:
    """The validity test this dataset makes possible.

    recovery_1 and recovery_2 are the SAME TASK as working_baseline. A decoder
    that has learned load should put all three at a similar level. One that
    has learned time-on-task, fatigue or electrode drift will show recovery
    creeping upward.

    Pass condition: each recovery sits closer to working_baseline than to the
    stress block that preceded it.
    """
    need = ["working_baseline", "stress_1", "recovery_1", "stress_2", "recovery_2"]
    missing = [p for p in need if p not in expected_by_phase]
    if missing:
        return {"ok": False, "reason": f"missing phases: {missing}"}

    wb = expected_by_phase["working_baseline"]
    out = {"working_baseline": wb, "checks": {}}
    passed = True

    for rec, stress in (("recovery_1", "stress_1"), ("recovery_2", "stress_2")):
        r = expected_by_phase[rec]
        s = expected_by_phase[stress]
        d_base = abs(r - wb)
        d_stress = abs(r - s)
        ok = d_base < d_stress
        passed &= ok
        out["checks"][rec] = {
            "estimate": r,
            "stress_estimate": s,
            "dist_to_baseline": d_base,
            "dist_to_stress": d_stress,
            "returns_to_baseline": bool(ok),
        }

    out["ok"] = bool(passed)
    return out
