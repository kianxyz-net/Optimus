# cogload

Ordinal cognitive-load decoder for EEG, built around the PhysioNet
stress-resilience dataset. Outputs a continuous load level with calibrated
per-level confidence, runs live on a BrainFlow or LSL headset, and ships with a
validity check that most workload demos quietly skip.

---

## The headline result is the recovery check, not the accuracy

The dataset's six-phase design gives you something rare: `recovery_1` and
`recovery_2` are **the same task** as `working_baseline`, just with a stress
block in between. So you can ask a question no accuracy number answers —

> does the meter come back *down* when the demand goes away?

A model that learned cognitive load says yes. A model that learned
time-on-task, fatigue, or electrode drift shows recovery creeping upward — and
it will still post a respectable accuracy while doing it. `train_stress.py`
prints this first, before any other metric, labelled PASS or FAIL.

Lead the demo with it. "Here's the meter going up under stress and back down in
recovery, on a subject the model never saw" beats any confusion matrix.

---

## Data

**Primary:** Neurophysiological Dataset of Stress Resilience During
Human-Computer Interaction
<https://physionet.org/content/neuro-stress-resilience-hci/1.0.0/>
(doi 10.13026/x3vc-p627, Open Data Commons ODbL, 19.1 GB / 4.6 GB zipped)

35 participants on MATB-II, 32-channel EEG at 250 Hz from a g.tec g.Nautilus,
plus synchronised fNIRS, eye tracking, EDA, BVP, HR. Six phases:

| phase | task | level |
|---|---|---|
| resting_baseline (1 min) | eyes open, nothing | 0 |
| working_baseline (6 min) | RESMAN only | 1 |
| stress_1 (6 min) | RESMAN + COMMS + pump failures | 2 |
| recovery_1 (6 min) | RESMAN only | 1 |
| stress_2 (6 min) | RESMAN + COMMS + pump failures | 2 |
| recovery_2 (6 min) | RESMAN only | 1 |

Two stress blocks per person also means you can train on the first half and
test on the second, within subject (`--within-subject`).

**Secondary, for a ten-minute sanity check:** STEW
<https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset>
(48 subjects, Emotiv EPOC, 14 ch, 52 MB). Its paper reports 69% on three
workload levels. Run `train.py --stew` once: if you clear that, your pipeline
works and you can trust whatever the bigger dataset gives you. Without a
reference number somewhere in your weekend you cannot tell "my code is broken"
from "this problem is hard."

---

## Install and run

```bash
pip install numpy scipy scikit-learn joblib mne brainflow
pip install pyedflib pylsl   # EEGMAT / LSL streaming

python train_stress.py --synthetic                    # no download needed
python train_stress.py --data /path/to/.../1.0.0      # the real thing
python train_stress.py --data ... --modality fnirs    # hemodynamic arm
python train_stress.py --data ... --within-subject    # stability over time

python live.py --board synthetic --baseline-sec 20
python live.py --board unicorn --serial-port COM3 --model stress_decoder.joblib
```

Synthetic mode builds data in the real phase structure with the documented
effect direction planted, and gives recovery phases the same generative
parameters as working baseline — so the recovery check is a genuine test of the
pipeline, not a formality. Expect MAE ~0.18 against a chance MAE of 0.50.

---

## File format notes (the fiddly part)

Each participant CSV has **74 rows and one column per sample** — rows are
signals, columns are time. That is already the orientation the feature code
wants, so no transpose.

```
row  1      timestamp
rows 2-17   fNIRS optical density       (16)
rows 18-33  fNIRS concentration HbO/HbR (16)
rows 34-65  EEG, 32 channels            (32)
rows 66-73  eye tracking                 (8)
row  74     event markers
```

**Channel-order warning.** The dataset docs list the 32 EEG channels in two
different orders — once as prose, once with explicit electrode numbers — and
they disagree at positions 4–7 and 20–22. `stress_hci.py` uses the numbered
order and keeps the prose order beside it so you can swap and compare. It
barely matters here: every disagreeing pair sits in the same scalp region, and
the extractor averages within region first. If your results *do* move when you
swap, you are over-relying on individual channels.

**Markers.** Five per subject: the first starts `working_baseline`, each
subsequent one starts the next phase. The docs warn that some subjects have
extra or missing markers. Default behaviour is to **skip** those subjects — a
mislabelled subject is worse than a missing one. `--loose-markers` falls back
to nominal protocol timing instead, and the loader reports how many subjects
needed it so you can state it honestly.

---

## How it works

**Features** (`features.py`) — relative band power per scalp region, plus
theta/alpha ratio, an engagement index, and spectral entropy. Everything is
relative, so amplifier gain and impedance cancel. Region averaging means a
32-channel g.Nautilus and a 14-channel Emotiv produce the same feature vector.

**Baseline normalisation** — every subject's features are z-scored against
their own resting baseline. This is what lets a model trained on 33 strangers
work on the 34th, and it's why the live tool demands 60 s of rest before it
will say anything.

**Ordinal model** (`ordinal.py`) — cumulative binary decomposition (Frank &
Hall): fit `P(y>0)` and `P(y>1)`, derive per-level probabilities, enforce
monotonicity. Each sub-model is binary, so sklearn's calibration applies
directly. The expected level `Σ k·P(y=k)` is a smooth scalar in [0, 2] — that's
the meter. A 3-class argmax would throw the ordering away and jump.

**Evaluation** — `GroupKFold` by subject throughout. Reports MAE in load levels
against the always-guess-majority baseline, because for an ordered target MAE
matches the user's experience: accuracy 0.6 with MAE 0.42 is a usable meter,
accuracy 0.6 with MAE 0.9 is not, and accuracy alone cannot tell them apart.

---

## The fNIRS angle

`--modality fnirs` runs the identical pipeline on rows 18–33, the HbO/HbR
concentrations over prefrontal cortex. That is cerebral hemodynamics — the same
physical quantity Openwater's Open-Motion measures as BFI/BVI. If you want to
connect to Track 1 or 3 without their data, this is the closest public proxy
that exists, and you can show EEG and hemodynamics side by side on the same
subject and the same phases.

Expect fNIRS to be slower and weaker than EEG: the hemodynamic response lags
several seconds, so use longer windows (`--win 10 --step 5`) and don't expect
it to track within-phase fluctuation.

---

## Build order

1. `python train_stress.py --synthetic` — confirm the plumbing. MAE ~0.18.
2. `python train.py --stew /path/to/STEW` — ten-minute reality check against a
   published number.
3. Download the stress dataset. Start with **three subjects**, not all 35:
   `--subjects p01 p02 p03`. Verify markers parse before committing.
4. Full run. **If the recovery check FAILs, stop and fix it** — nothing
   downstream is meaningful until it passes.
5. `--within-subject`, then `--modality fnirs` if time remains.
6. Live headset. Budget two hours; it always takes two hours.

---

## What to say, and what not to

**Say:** "Three-level load estimate with calibrated confidence, validated by
showing the estimate returns to baseline during recovery, evaluated on held-out
subjects."

**Don't say:** "Detects sensory overload in autistic people." The stressor is an
added task plus induced failures, so level 2 is high load *and* high arousal —
call it load/stress. Participants are neurotypical adults doing a
flight-sim-style task. There is no validated EEG biomarker for autistic sensory
overload, and a confident wrong readout aimed at someone already overwhelmed is
worse than no readout.

What you *can* offer an autistic user is a meter they calibrate and label
themselves. `live.py --log` writes timestamped readings; add a keypress for "too
much right now" and you have per-person ground truth — more defensible and more
useful than a population model.

---

## Known limitations for the last slide

- Level 2 confounds cognitive load with arousal; the design doesn't separate them.
- No artifact rejection beyond linear detrend. Blinks and jaw tension sit in the
  same bands as the signal.
- 35 participants, mean age 26 — exploratory scale, not population-representative.
- Calibration is fitted on the training population; stated confidences will drift
  on a different headset.
- `recovery_check` tests direction, not magnitude. It rules out the obvious
  confound, not every confound.
- The dataset is new (published Feb 2026, v1.0.0) with no published benchmark, so
  there is no external number to check yourself against. That's what STEW is for.
