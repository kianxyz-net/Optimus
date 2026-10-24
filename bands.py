"""
Species-specific frequency band definitions.

This is the only place where species differences are hard-coded. Everything
downstream works on band-relative features, so the rest of the pipeline is
species-agnostic.

Why band edges differ by species: rodent cortical oscillations are not a
frequency-shifted copy of human ones. Rodent "theta" (hippocampal, ~6-10 Hz)
overlaps the human alpha range but is a different phenomenon, and rodents have
no clear posterior alpha rhythm at all. So we do NOT claim a 1:1 band mapping.
What we claim is narrower and defensible:

    the SHAPE of the spectrum shifts in a measurable way under task load,
    and the same feature extractor can quantify that shift in either species.

Human band edges: conventional clinical ranges.
Mouse band edges: common rodent EEG ranges from the surface-EEG literature.
Macaque band edges: close to human; primate cortex is the reason monkey work
transfers better than rodent work.
"""

from __future__ import annotations

# name -> (low_hz, high_hz)
HUMAN_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

MACAQUE_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta": (15.0, 30.0),
    "gamma": (30.0, 50.0),
}

MOUSE_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (5.0, 10.0),
    "alpha": (10.0, 15.0),  # no true rodent alpha; kept as a spectral slot only
    "beta": (15.0, 30.0),
    "gamma": (30.0, 80.0),
}

BANDS_BY_SPECIES = {
    "human": HUMAN_BANDS,
    "macaque": MACAQUE_BANDS,
    "mouse": MOUSE_BANDS,
}

# Rough scalp-region grouping for human montages. Used to build region-averaged
# features so that a 14-channel Emotiv and a 23-channel Neurocom cap produce the
# same feature vector length.
HUMAN_REGIONS = {
    "frontal": [
        "AF3", "AF4", "AF7", "AF8", "AFz",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz",
        "Fp1", "Fp2", "Fpz",
        "FT7", "FT8",
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
    ],
    "central": [
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz",
        "T7", "T8", "T3", "T4",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
    ],
    "posterior": [
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz",
        "PO3", "PO4", "PO7", "PO8", "POz",
        "O1", "O2", "Oz",
        "T5", "T6",
    ],
}


def bands_for(species: str) -> dict:
    key = species.lower()
    if key not in BANDS_BY_SPECIES:
        raise ValueError(
            f"unknown species {species!r}; expected one of {sorted(BANDS_BY_SPECIES)}"
        )
    return BANDS_BY_SPECIES[key]


def region_of(channel_name: str) -> str | None:
    """Map a 10-20 channel label to a coarse region, or None if unrecognised."""
    name = channel_name.strip().upper()
    for region, members in HUMAN_REGIONS.items():
        if name in {m.upper() for m in members}:
            return region
    return None
