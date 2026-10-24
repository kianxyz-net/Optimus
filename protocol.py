"""
The MATB-II stress-resilience protocol, and the load levels it defines.

Six phases, one minute then five six-minute blocks:

    resting_baseline   eyes open, no task                  level 0
    working_baseline   RESMAN only                         level 1
    stress_1           RESMAN + COMMS + pump failures      level 2
    recovery_1         RESMAN only                         level 1
    stress_2           RESMAN + COMMS + pump failures      level 2
    recovery_2         RESMAN only                         level 1

Two design facts make this dataset worth the extra parsing work:

1. RECOVERY REPEATS WORKING BASELINE. Same task, same demand, different
   history. So "does the model put recovery back at level 1?" is a free
   validity check that needs no extra labels. If your decoder calls recovery
   high-load, it has learned time-on-task or electrode drift, not load.

2. TWO INDEPENDENT STRESS BLOCKS PER PERSON. Train on stress_1, test on
   stress_2, within subject. Nothing else you have access to supports that.

A caution on naming: the stressor is an added task plus induced failures, so
level 2 is high load AND high arousal. Call the output "load/stress". Calling
it pure cognitive load overstates what the design isolates.
"""

from __future__ import annotations

# name, duration in minutes, ordinal load level
PHASES = [
    ("resting_baseline", 1, 0),
    ("working_baseline", 6, 1),
    ("stress_1", 6, 2),
    ("recovery_1", 6, 1),
    ("stress_2", 6, 2),
    ("recovery_2", 6, 1),
]

PHASE_NAMES = [p[0] for p in PHASES]
PHASE_LEVEL = {p[0]: p[2] for p in PHASES}
PHASE_MINUTES = {p[0]: p[1] for p in PHASES}

LEVEL_NAMES = {0: "rest", 1: "task", 2: "high load"}
N_LEVELS = 3

# Markers are sent at the START of working_baseline and every 6 min after,
# so the final marker starts recovery_2. Five markers for six phases: the
# resting baseline is whatever precedes the first marker.
EXPECTED_MARKERS = 5
MARKER_PHASES = PHASE_NAMES[1:]  # phases that a marker begins


def expected_boundaries_sec() -> list[tuple[str, float, float]]:
    """Nominal (phase, start_sec, end_sec) if the protocol ran exactly to time.

    Used as a fallback when a subject's markers are missing or malformed,
    which the dataset documentation warns happens. Always prefer real markers;
    this is for salvaging a subject rather than dropping it.
    """
    out, t = [], 0.0
    for name, minutes, _ in PHASES:
        out.append((name, t, t + minutes * 60.0))
        t += minutes * 60.0
    return out
