#!/usr/bin/env python3
"""
Train the ordinal load decoder on the stress-resilience dataset.

    python train_stress.py --synthetic                  # no download needed
    python train_stress.py --data /path/to/neuro-stress-resilience-hci/1.0.0
    python train_stress.py --data ... --modality fnirs  # hemodynamic arm
    python train_stress.py --data ... --within-subject  # train S1, test S2

Outputs, in order of how much they matter:

  1. RECOVERY CHECK. Does the model put recovery back near working baseline?
     If not, it learned time-on-task and nothing else is worth reading.
  2. MAE in load levels, against the always-guess-majority baseline.
  3. Accuracy and the confusion matrix.

Accuracy is listed third on purpose. For an ordered target it is the least
informative of the three.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cogload.features import extract_recording, BaselineNormalizer
from cogload.ordinal import OrdinalLoadDecoder, evaluate_ordinal, recovery_check
from cogload.protocol import PHASE_NAMES, PHASE_LEVEL, LEVEL_NAMES, N_LEVELS
from cogload.stress_hci import load_stress_hci, SFREQ, EEG_CHANNELS


def featurize(recordings, win_sec=4.0, step_sec=2.0):
    """Per-phase Recordings -> (X, y, groups, phases, names).

    Window length defaults to 4 s rather than the 2 s used for the binary
    decoder. Phases here are six minutes long, so there is no shortage of
    data, and a longer window gives a cleaner spectral estimate at the low
    frequencies that carry the load signal.
    """
    by_subject: dict = {}
    names = None

    for rec in recordings:
        try:
            X, names = extract_recording(
                rec.data, rec.sfreq, rec.ch_names,
                species=rec.species, win_sec=win_sec, step_sec=step_sec,
            )
        except ValueError:
            continue  # phase shorter than a window
        phase = rec.meta.get("phase", rec.condition)
        by_subject.setdefault(rec.subject, {})[phase] = (X, rec.label)

    Xs, ys, gs, ps = [], [], [], []
    skipped = 0

    for subject, phases in by_subject.items():
        if "resting_baseline" not in phases:
            skipped += 1
            continue
        norm = BaselineNormalizer().fit(phases["resting_baseline"][0])

        for phase, (X, label) in phases.items():
            Z = norm.transform(X)
            Xs.append(Z)
            ys.append(np.full(len(Z), label, dtype=int))
            gs.append(np.full(len(Z), subject, dtype=object))
            ps.append(np.full(len(Z), phase, dtype=object))

    if not Xs:
        raise SystemExit("no usable subjects -- check the data path")
    if skipped:
        print(f"  skipped {skipped} subject(s) with no resting baseline")

    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(ps), names)


def make_synthetic(n_subjects=24, seconds_per_phase=90, seed=0):
    """Synthetic data in the real phase structure.

    Plants the documented direction of the effect -- frontal theta up and
    posterior alpha down with load -- and gives every subject its own
    amplitude and peak frequency so the baseline normaliser has real work to
    do. Crucially, recovery phases get the SAME generative parameters as
    working baseline, so the recovery check is a genuine test of the pipeline
    rather than a formality.
    """
    from cogload.datasets import Recording

    rng = np.random.default_rng(seed)
    n = int(SFREQ * seconds_per_phase)
    t = np.arange(n) / SFREQ
    recs = []

    for s in range(n_subjects):
        scale = rng.uniform(0.5, 3.0)
        alpha_f = rng.uniform(8.5, 12.0)
        theta_f = rng.uniform(4.5, 7.5)
        sens = rng.uniform(0.6, 1.4)  # how strongly this person responds

        for phase in PHASE_NAMES:
            level = PHASE_LEVEL[phase]
            alpha_amp = max(0.1, (1.1 - 0.28 * level * sens) * rng.uniform(0.85, 1.15))
            theta_amp = max(0.1, (0.35 + 0.30 * level * sens) * rng.uniform(0.85, 1.15))

            chans = []
            for ch in EEG_CHANNELS:
                x = rng.normal(0, 1.0, n)
                x = np.cumsum(x) / np.sqrt(n)      # 1/f-ish background
                x -= x.mean()
                up = ch.upper()
                posterior = up.startswith(("P", "O"))
                frontal = up.startswith(("AF", "F"))
                if posterior:
                    x += alpha_amp * 0.9 * np.sin(2 * np.pi * alpha_f * t + rng.uniform(0, 7))
                if frontal:
                    x += theta_amp * 0.9 * np.sin(2 * np.pi * theta_f * t + rng.uniform(0, 7))
                chans.append(x * scale)

            recs.append(
                Recording(
                    data=np.asarray(chans),
                    sfreq=SFREQ,
                    ch_names=list(EEG_CHANNELS),
                    subject=f"syn-{s:02d}",
                    condition="rest" if phase == "resting_baseline" else phase,
                    label=level,
                    meta={"dataset": "synthetic", "phase": phase},
                )
            )
    return recs


def phase_means(expected, phases) -> dict:
    return {
        p: float(np.mean(expected[phases == p]))
        for p in PHASE_NAMES
        if (phases == p).any()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="root of the stress-hci dataset")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--modality", default="eeg", choices=["eeg", "fnirs"])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    ap.add_argument("--loose-markers", action="store_true",
                    help="fall back to nominal timing when markers don't parse")
    ap.add_argument("--within-subject", action="store_true",
                    help="also train on stress_1 and test on stress_2")
    ap.add_argument("--out", default="stress_decoder.joblib")
    args = ap.parse_args()

    if not (args.data or args.synthetic):
        ap.error("give --data or --synthetic")

    if args.synthetic:
        print("building synthetic data in the real phase structure...")
        recs = make_synthetic()
    else:
        print(f"loading from {args.data} (modality={args.modality})...")
        recs = load_stress_hci(
            args.data,
            subjects=args.subjects,
            modality=args.modality,
            strict_markers=not args.loose_markers,
        )

    print(f"{len(recs)} phase-recordings")
    X, y, groups, phases, names = featurize(recs, args.win, args.step)
    print(f"  X {X.shape}, {len(names)} features, "
          f"{len(np.unique(groups))} subjects")
    print(f"  level counts: " + ", ".join(
        f"{LEVEL_NAMES[k]}={int((y == k).sum())}" for k in range(N_LEVELS)))

    res = evaluate_ordinal(X, y, groups, kind=args.model)

    # ---------------- recovery check: the headline ----------------
    pm = phase_means(res["oof_expected"], phases)
    rc = recovery_check(pm)

    print(f"\n{'=' * 64}")
    print("RECOVERY CHECK  (does the meter come back down?)")
    print("=" * 64)
    print("  mean estimated level per phase, held-out subjects:")
    for p in PHASE_NAMES:
        if p in pm:
            bar = "#" * int(round(pm[p] * 18))
            print(f"    {p:<20}{pm[p]:5.2f}  (true {PHASE_LEVEL[p]})  {bar}")
    if rc.get("ok") is not None:
        for phase, c in rc.get("checks", {}).items():
            mark = "PASS" if c["returns_to_baseline"] else "FAIL"
            print(f"    {phase}: {mark}  "
                  f"dist to baseline {c['dist_to_baseline']:.2f} vs "
                  f"to stress {c['dist_to_stress']:.2f}")
        print(f"\n  overall: {'PASS' if rc['ok'] else 'FAIL'}")
        if not rc["ok"]:
            print("  A FAIL here means the model tracks time-on-task or drift,")
            print("  not load. Fix that before trusting anything below.")

    # ---------------- ordinal metrics ----------------
    print(f"\n{'=' * 64}")
    print("SUBJECT-WISE CROSS-VALIDATION (unseen people)")
    print("=" * 64)
    print(f"  windows              {res['n_windows']}")
    print(f"  subjects             {res['n_subjects']}")
    print(f"  MAE (load levels)    {res['mae']:.3f}")
    print(f"  chance MAE           {res['chance_mae']:.3f}   <- must beat this")
    print(f"  accuracy             {res['accuracy']:.3f}")
    print(f"  majority baseline    {res['majority_baseline']:.3f}")
    print(f"  within-1-level       {res['adjacent_accuracy']:.3f}")
    print(f"  coverage @conf .50   {res['coverage_at_conf_50']:.1%}")
    print(f"  accuracy when conf   {res['accuracy_when_confident']:.3f}")

    print("\n  confusion (rows true, cols predicted)")
    header = "".join(f"{LEVEL_NAMES[k]:>12}" for k in range(N_LEVELS))
    print(f"    {'':<12}{header}")
    for k, row in enumerate(res["confusion_matrix"]):
        cells = "".join(f"{v:>12d}" for v in row)
        print(f"    {LEVEL_NAMES[k]:<12}{cells}")

    print("\n  per level")
    for name, d in res["per_level"].items():
        print(f"    {name:<12} n={d['n']:<7} recall={d['recall']:.3f}  "
              f"mean estimate={d['mean_estimate']:.2f}")

    # ---------------- final model ----------------
    dec = OrdinalLoadDecoder(kind=args.model).fit(
        X, y, groups=groups, feature_names=names
    )
    coefs = dec.coefficients(threshold=1)
    if coefs:
        print("\n  top features for the high-load boundary")
        for k, v in list(coefs.items())[:8]:
            print(f"    {k:<24}{v:+.3f}")

    # ---------------- within-subject arm ----------------
    if args.within_subject:
        tr = np.isin(phases, ["resting_baseline", "working_baseline", "stress_1"])
        te = np.isin(phases, ["recovery_1", "stress_2"])
        if tr.any() and te.any():
            d2 = OrdinalLoadDecoder(kind=args.model).fit(
                X[tr], y[tr], groups=groups[tr]
            )
            pred = d2.predict_level(X[te])
            mae = float(np.mean(np.abs(pred - y[te])))
            print(f"\n{'=' * 64}")
            print("WITHIN-SUBJECT: train on first half, test on second half")
            print("=" * 64)
            print(f"  same people, later blocks -- tests stability over time")
            print(f"  accuracy  {np.mean(pred == y[te]):.3f}")
            print(f"  MAE       {mae:.3f}")

    try:
        import joblib
        joblib.dump(
            {"decoder": dec, "feature_names": names,
             "modality": args.modality, "n_levels": N_LEVELS},
            args.out,
        )
        print(f"\nsaved -> {args.out}")
    except ImportError:
        print("\n(install joblib to save)")

    summary = {k: v for k, v in res.items()
               if k not in ("oof_expected", "oof_level")}
    summary["phase_means"] = pm
    summary["recovery_check"] = {k: v for k, v in rc.items()}
    with open("stress_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("saved -> stress_results.json")


if __name__ == "__main__":
    main()
