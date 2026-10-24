#!/usr/bin/env python3
"""
Train and evaluate the cognitive-load decoder.

    python train.py --stew /path/to/STEW
    python train.py --stew /path/to/STEW --eegmat /path/to/eegmat --cross-dataset
    python train.py --synthetic            # no data needed, checks the pipeline

Always reports the majority-class baseline next to accuracy. If your accuracy
is not clearly above it, you have nothing, and it is better to find that out
on Saturday than during judging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from cogload.datasets import load_stew, load_eegmat
from cogload.features import extract_recording, BaselineNormalizer
from cogload.model import LoadDecoder, evaluate
from cogload.crossspecies import coral


def featurize(recordings, win_sec=2.0, step_sec=1.0, per_subject_baseline=True):
    """Recordings -> (X, y, groups, feature_names).

    With per_subject_baseline, each subject's windows are z-scored against
    their own rest block. This is what lets a model trained on 48 strangers
    work on a 49th person. Without it, between-subject variance swamps the
    load effect and cross-subject accuracy collapses toward chance.
    """
    by_subject: dict = {}
    names = None

    for rec in recordings:
        X, names = extract_recording(
            rec.data, rec.sfreq, rec.ch_names,
            species=rec.species, win_sec=win_sec, step_sec=step_sec,
        )
        by_subject.setdefault(rec.subject, {})[rec.condition] = (X, rec.label)

    Xs, ys, gs = [], [], []
    skipped = 0
    for subject, conds in by_subject.items():
        if per_subject_baseline:
            if "rest" not in conds:
                skipped += 1
                continue
            norm = BaselineNormalizer().fit(conds["rest"][0])
        else:
            norm = None

        for cond, (X, label) in conds.items():
            if label is None:
                continue
            Z = norm.transform(X) if norm is not None else X
            Xs.append(Z)
            ys.append(np.full(len(Z), label))
            gs.append(np.full(len(Z), subject, dtype=object))

    if not Xs:
        raise SystemExit("no usable recordings -- check the data path")
    if skipped:
        print(f"  skipped {skipped} subject(s) with no rest block", file=sys.stderr)

    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), names


def make_synthetic(n_subjects=20, sfreq=128.0, seconds=120, seed=0):
    """Synthetic EEG with a planted load effect, for testing the plumbing.

    Rest has a strong posterior alpha peak. Task has reduced alpha and raised
    frontal theta -- the real direction of the effect. Each subject gets its
    own amplitude scale and alpha frequency so the per-subject normalisation
    has something to actually remove.
    """
    from cogload.datasets import Recording, STEW_CHANNELS

    rng = np.random.default_rng(seed)
    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    recs = []

    for s in range(n_subjects):
        scale = rng.uniform(0.5, 3.0)          # per-subject amplitude
        alpha_f = rng.uniform(9.0, 12.0)       # per-subject alpha peak
        theta_f = rng.uniform(5.0, 7.0)

        # Effect sizes are deliberately small and overlapping between subjects.
        # A synthetic test that comes out at 100% tells you nothing about
        # whether the calibration and abstention logic work; this lands in the
        # 0.7-0.9 range where real EEG workload decoding actually lives.
        for cond, label in (("rest", 0), ("task", 1)):
            alpha_amp = rng.uniform(0.7, 1.3) if cond == "rest" else rng.uniform(0.4, 1.0)
            theta_amp = rng.uniform(0.3, 0.9) if cond == "rest" else rng.uniform(0.5, 1.1)

            chans = []
            for ch in STEW_CHANNELS:
                x = rng.normal(0, 1.0, n)  # broadband background
                posterior = ch[0] in ("O", "P")
                frontal = ch[0] in ("A", "F")
                if posterior:
                    x += alpha_amp * 0.9 * np.sin(2 * np.pi * alpha_f * t + rng.uniform(0, 7))
                if frontal:
                    x += theta_amp * 0.9 * np.sin(2 * np.pi * theta_f * t + rng.uniform(0, 7))
                chans.append(x * scale)

            recs.append(
                Recording(
                    data=np.array(chans),
                    sfreq=sfreq,
                    ch_names=list(STEW_CHANNELS),
                    subject=f"syn-{s}",
                    condition=cond,
                    label=label,
                    meta={"dataset": "synthetic"},
                )
            )
    return recs


def report(res, title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
    print(f"  windows            {res['n_windows']}")
    print(f"  subjects           {res['n_subjects']}")
    print(f"  accuracy           {res['accuracy']:.3f}")
    print(f"  majority baseline  {res['majority_baseline']:.3f}   <- must beat this")
    print(f"  AUC                {res['auc']:.3f}")
    cm = res["confusion_matrix"]
    print(f"  confusion          [[{cm[0][0]:5d} {cm[0][1]:5d}]")
    print(f"                      [{cm[1][0]:5d} {cm[1][1]:5d}]]")

    ab = res["abstention"]
    print(f"\n  with abstention band 0.35-0.65:")
    print(f"    answers given    {ab['coverage']:.1%} of windows")
    print(f"    accuracy on those {ab['accuracy_when_confident']:.3f}")

    print("\n  calibration (is the confidence honest?)")
    print(f"    {'bin':<12}{'n':>7}{'stated':>9}{'actual':>9}")
    for row in res["reliability"]:
        print(f"    {row['bin']:<12}{row['n']:>7}{row['stated']:>9.2f}{row['actual']:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stew", help="directory of STEW .txt files")
    ap.add_argument("--eegmat", help="directory of EEGMAT .edf files")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data")
    ap.add_argument("--cross-dataset", action="store_true",
                    help="train on STEW, test on EEGMAT")
    ap.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    ap.add_argument("--win", type=float, default=2.0, help="window seconds")
    ap.add_argument("--step", type=float, default=1.0, help="step seconds")
    ap.add_argument("--out", default="decoder.joblib")
    args = ap.parse_args()

    if not (args.stew or args.eegmat or args.synthetic):
        ap.error("give --stew, --eegmat or --synthetic")

    recs = []
    if args.synthetic:
        print("loading synthetic data...")
        recs += make_synthetic()
    if args.stew:
        print(f"loading STEW from {args.stew} ...")
        recs += load_stew(args.stew)
    if args.eegmat and not args.cross_dataset:
        print(f"loading EEGMAT from {args.eegmat} ...")
        recs += load_eegmat(args.eegmat)

    print(f"{len(recs)} recordings")
    print("extracting features...")
    X, y, groups, names = featurize(X_in := recs, win_sec=args.win, step_sec=args.step)
    print(f"  X {X.shape}  features: {len(names)}")

    res = evaluate(X, y, groups, kind=args.model)
    report(res, "SUBJECT-WISE CROSS-VALIDATION (unseen people)")

    # final model on everything
    dec = LoadDecoder(kind=args.model).fit(X, y, groups=groups, feature_names=names)
    coefs = dec.coefficients()
    if coefs:
        print("\n  top features (logistic weights, + means predicts high load)")
        for k, v in list(coefs.items())[:8]:
            print(f"    {k:<24}{v:+.3f}")

    if args.cross_dataset:
        if not (args.stew and args.eegmat):
            ap.error("--cross-dataset needs both --stew and --eegmat")
        print(f"\nloading EEGMAT as held-out domain...")
        te_recs = load_eegmat(args.eegmat)
        Xt, yt, gt, _ = featurize(te_recs, win_sec=args.win, step_sec=args.step)

        p_raw = dec.predict_load(Xt)
        acc_raw = np.mean((p_raw >= 0.5).astype(int) == yt)

        Xs_al = coral(X, Xt)
        dec_al = LoadDecoder(kind=args.model).fit(Xs_al, y, groups=groups,
                                                  feature_names=names)
        p_al = dec_al.predict_load(Xt)
        acc_al = np.mean((p_al >= 0.5).astype(int) == yt)

        print(f"\n{'=' * 62}\nCROSS-DATASET: train STEW -> test EEGMAT\n{'=' * 62}")
        print(f"  different hardware, different task, different country")
        print(f"  accuracy, no alignment    {acc_raw:.3f}")
        print(f"  accuracy, CORAL aligned   {acc_al:.3f}")
        print(f"  majority baseline         {max(np.mean(yt==0), np.mean(yt==1)):.3f}")
        print("\n  This is the number to quote. Subject-wise CV within one")
        print("  dataset still shares hardware and task; this does not.")

    try:
        import joblib
        joblib.dump({"decoder": dec, "feature_names": names}, args.out)
        print(f"\nsaved -> {args.out}")
    except ImportError:
        print("\n(install joblib to save the model)")

    summary = {k: v for k, v in res.items() if k != "oof_proba"}
    with open("results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("saved -> results.json")


if __name__ == "__main__":
    main()
