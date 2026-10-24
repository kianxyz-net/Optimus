#!/usr/bin/env python3
"""
Cross-species validation of the feature extractor.

    python crossspecies_demo.py --synthetic
    python crossspecies_demo.py --mouse-baseline rest.set --mouse-active stim.set \
                                --human-baseline sub01_lo.txt --human-active sub01_hi.txt

The claim this script supports, stated precisely:

    The same feature extractor, run unchanged on mouse and human EEG, detects
    a spectral shift between baseline and active conditions, and a subset of
    features shifts in the same direction in both species.

The claim it does NOT support, and which you should refuse to make if a judge
pushes you toward it:

    "We trained on animal brains and decoded human cognitive load."

Nobody has labelled cognitive load in an animal EEG set, mouse and human
cortex do not share an electrode correspondence, and a model fitted on one
and applied to the other is a random number generator with good UI. The
consistency analysis below is the real, smaller, defensible version.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cogload.crossspecies import spectral_shift, compare_species


def synth(sfreq, seconds, n_ch, peak_hz, peak_amp, seed, ch_names=None):
    """Narrowband oscillation plus 1/f-ish background."""
    rng = np.random.default_rng(seed)
    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    rows = []
    for _ in range(n_ch):
        x = rng.normal(0, 1, n)
        # crude 1/f: cumulative sum then detrend-ish
        x = np.cumsum(x) / np.sqrt(n)
        x = x - x.mean()
        x += peak_amp * np.sin(2 * np.pi * peak_hz * t + rng.uniform(0, 7))
        rows.append(x)
    return np.asarray(rows)


def run_synthetic():
    """Both species: dominant resting rhythm weakens under 'load'.

    Human: posterior alpha ~10 Hz drops.
    Mouse: theta ~7 Hz drops, at a different sampling rate and channel count.
    If the extractor is doing its job, the relative-power features move in the
    same direction despite everything else differing.
    """
    results = []

    human_names = ["AF3", "F3", "F4", "AF4", "P7", "P8", "O1", "O2"]
    results.append(
        spectral_shift(
            baseline=synth(128, 120, 8, 10.0, 1.2, seed=1),
            active=synth(128, 120, 8, 10.0, 0.4, seed=2),
            sfreq=128.0,
            species="human",
            ch_names=human_names,
        )
    )

    results.append(
        spectral_shift(
            baseline=synth(1000, 120, 4, 7.0, 1.2, seed=3),
            active=synth(1000, 120, 4, 7.0, 0.4, seed=4),
            sfreq=1000.0,
            species="mouse",
            ch_names=["m1", "m2", "m3", "m4"],
        )
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--human-baseline")
    ap.add_argument("--human-active")
    ap.add_argument("--mouse-baseline")
    ap.add_argument("--mouse-active")
    ap.add_argument("--macaque-baseline")
    ap.add_argument("--macaque-active")
    ap.add_argument("--out", default="crossspecies.json")
    args = ap.parse_args()

    if args.synthetic:
        results = run_synthetic()
    else:
        from cogload.datasets import load_animal
        import numpy as np

        results = []

        if args.human_baseline and args.human_active:
            from cogload.datasets import STEW_CHANNELS, STEW_SFREQ
            b = np.loadtxt(args.human_baseline).T
            a = np.loadtxt(args.human_active).T
            results.append(
                spectral_shift(b, a, STEW_SFREQ, "human", list(STEW_CHANNELS))
            )

        for species, bpath, apath in (
            ("mouse", args.mouse_baseline, args.mouse_active),
            ("macaque", args.macaque_baseline, args.macaque_active),
        ):
            if bpath and apath:
                rb = load_animal(bpath, species=species)
                ra = load_animal(apath, species=species)
                results.append(
                    spectral_shift(rb.data, ra.data, rb.sfreq, species, rb.ch_names)
                )

        if not results:
            ap.error("give --synthetic or at least one baseline/active pair")

    for r in results:
        print(f"\n{'=' * 62}")
        print(f"{r['species'].upper()}  "
              f"({r['n_baseline_windows']} baseline / {r['n_active_windows']} active windows)")
        print("=" * 62)
        print("  effect size of each feature (Cohen's d, baseline -> active)")
        for feat, d in list(r["cohens_d"].items())[:8]:
            flag = "  <- large" if abs(d) > 0.8 else ""
            print(f"    {feat:<24}{d:+.2f}{flag}")

    if len(results) > 1:
        cmp = compare_species(results)
        print(f"\n{'=' * 62}")
        print("CROSS-SPECIES CONSISTENCY")
        print("=" * 62)
        print(f"  species compared: {', '.join(cmp['species'])}")
        print(f"  features moving the same direction in all: {cmp['n_consistent']}")
        for feat in cmp["consistent_features"]:
            per = cmp["features"][feat]["per_species"]
            vals = "  ".join(f"{k}={v:+.2f}" for k, v in per.items())
            print(f"    {feat:<24}{vals}")
        if not cmp["consistent_features"]:
            print("    none -- say so. A null result reported honestly is a")
            print("    result; a fabricated agreement is not.")

        with open(args.out, "w") as fh:
            json.dump(cmp, fh, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
