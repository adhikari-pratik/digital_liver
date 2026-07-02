"""
Derived clinical readouts: fields the brief says are NOT stored/predicted but COMPUTED from the
state, so they can never disagree with it.

Cirrhosis is the brief's own example: "derived from fibrosis by a fixed monotone function and
never stored, so it can never disagree with F." We predict F (monotone by construction), so the
cirrhosis stage is a pure readout `g(F)` that inherits F's monotonicity for free -- it can never
regress and can never contradict fibrosis. Same principle as coupling M to F*C in the head.
"""

import numpy as np

CIRRHOSIS_F = 0.8   # fibrosis level at which the patient is cirrhotic (METAVIR F4-ish)


def cirrhosis_stage(F):
    """Fixed monotone map F in [0,1] -> METAVIR-like fibrosis stage 0..4 (4 = cirrhosis)."""
    return np.clip((np.asarray(F) / 0.2).astype(int), 0, 4)


def is_cirrhotic(F):
    return np.asarray(F) >= CIRRHOSIS_F


def cirrhosis_onset_month(F_traj):
    """First month F crosses the cirrhosis threshold; -1 if it never does."""
    idx = np.where(np.asarray(F_traj) >= CIRRHOSIS_F)[0]
    return int(idx[0]) if len(idx) else -1
