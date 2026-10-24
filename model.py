"""
The decoder.

Two things matter more than the classifier choice:

1. SUBJECT-WISE evaluation. EEG windows from the same person are massively
   correlated. If you shuffle windows into train/test you will see 95%+ and it
   will mean nothing. Every split here is grouped by subject, so the reported
   number answers "how well does this work on a person it has never seen",
   which is the only question the demo actually poses.

2. CALIBRATED confidence. A raw classifier score is not a probability. This
   wraps the model in Platt scaling / isotonic regression so that windows
   reported at 0.8 are right about 80% of the time. The confidence number is
   the deliverable here, so it has to be honest -- there is a reliability-curve
   check in evaluate() that tells you whether it is.

The classifier is a small regularised one on purpose. With a few thousand
windows and ~20 features, logistic regression calibrates better and overfits
less than a gradient-boosted anything, and you can read the coefficients out
loud to a judge.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(kind: str = "logreg", seed: int = 0):
    if kind == "logreg":
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=seed)),
            ]
        )
    elif kind == "rf":
        base = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return base


class LoadDecoder:
    """Binary cognitive-load decoder with calibrated probability output."""

    def __init__(self, kind: str = "logreg", calibration: str = "sigmoid", seed: int = 0):
        self.kind = kind
        self.calibration = calibration
        self.seed = seed
        self.model_ = None
        self.feature_names_: list[str] | None = None

    def fit(self, X, y, groups=None, feature_names=None) -> "LoadDecoder":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        base = build_model(self.kind, self.seed)
        n_groups = len(np.unique(groups)) if groups is not None else 0

        # CalibratedClassifierCV does not route a `groups` kwarg down to its
        # inner splitter, so materialise the group-wise splits ourselves and
        # hand them over as an explicit list. Without this the calibration
        # folds leak windows from the same subject across train and test, and
        # the resulting probabilities come out overconfident.
        if n_groups >= 2:
            n_splits = min(5, n_groups)
            cv = list(GroupKFold(n_splits=n_splits).split(X, y, np.asarray(groups)))
        else:
            cv = 3

        self.model_ = CalibratedClassifierCV(base, method=self.calibration, cv=cv)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit the decoder first")
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict_load(self, X) -> np.ndarray:
        """P(high load) for each window, in [0, 1]."""
        return self.predict_proba(X)[:, 1]

    def coefficients(self) -> dict | None:
        """Average logistic-regression weights across calibration folds."""
        if self.model_ is None or self.kind != "logreg":
            return None
        coefs = []
        for cc in self.model_.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                continue
            coefs.append(est.named_steps["clf"].coef_.ravel())
        if not coefs:
            return None
        mean = np.mean(coefs, axis=0)
        names = self.feature_names_ or [f"f{i}" for i in range(len(mean))]
        return dict(sorted(zip(names, mean), key=lambda kv: -abs(kv[1])))


def smooth_probabilities(p: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential smoothing over consecutive windows.

    Single-window EEG classification is jumpy. In a live readout you want the
    displayed value to move on the timescale of an actual mental state, not on
    the timescale of a blink. alpha=0.3 over 1 s steps gives roughly a 3 s
    effective time constant.
    """
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    acc = p[0]
    for i, v in enumerate(p):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def decide(p: float, low: float = 0.35, high: float = 0.65) -> str:
    """Three-way decision with an explicit abstention band.

    A two-class forced choice on a noisy signal produces confident nonsense
    near the boundary. Reporting "uncertain" there is both more accurate and,
    for anything a person might act on, the responsible default.
    """
    if p >= high:
        return "high load"
    if p <= low:
        return "low load"
    return "uncertain"


def evaluate(X, y, groups, kind: str = "logreg", calibration: str = "sigmoid",
             n_splits: int = 5, seed: int = 0) -> dict:
    """Subject-wise cross-validation with calibration quality.

    Returns accuracy, AUC, confusion matrix, a reliability curve, and the
    majority-class baseline you must beat for the result to mean anything.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("need at least 2 subjects for subject-wise CV")

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        dec = LoadDecoder(kind=kind, calibration=calibration, seed=seed)
        dec.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        oof[test_idx] = dec.predict_load(X[test_idx])

    pred = (oof >= 0.5).astype(int)
    majority = max(np.mean(y == 0), np.mean(y == 1))

    # reliability curve: are the stated confidences true?
    edges = np.linspace(0, 1, 11)
    reliability = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (oof >= lo) & (oof < hi)
        if m.sum() >= 10:
            reliability.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(m.sum()),
                    "stated": float(oof[m].mean()),
                    "actual": float(y[m].mean()),
                }
            )

    # how much is bought by refusing to answer near the boundary
    confident = (oof >= 0.65) | (oof <= 0.35)
    abstain_stats = {
        "coverage": float(confident.mean()),
        "accuracy_when_confident": (
            float(accuracy_score(y[confident], pred[confident]))
            if confident.any()
            else float("nan")
        ),
    }

    return {
        "n_windows": int(len(y)),
        "n_subjects": int(n_groups),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
        "majority_baseline": float(majority),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "reliability": reliability,
        "abstention": abstain_stats,
        "oof_proba": oof,
    }
