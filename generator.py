"""
Synthetic generator for the Digital Liver world model.

Emits monthly trajectories of an 8-D clinical state x(t). This generator is BOTH our
training data and our quality bar: a model that generalises across held-out patients has,
in effect, recovered these update rules. We keep the rules simple, seeded, and readable so
the "did it really learn the dynamics?" question has a concrete answer to compare against.

State layout x(t) in R^8, monthly timesteps. All fields in [0,1] except M in [0,2].

    idx  field  meaning                     temporal behaviour
    0    F      fibrosis                     ratchet, non-decreasing
    1    D      ductopenia (duct loss)       ratchet, irreversible (non-decreasing)
    2    S      biliary strictures           non-decreasing, steps DOWN at an ERCP event  (see D1)
    3    P      portal hypertension          ratchet, non-decreasing
    4    A      inflammatory activity        fast, mean-reverting
    5    C      cholestasis                  fast, with flares
    6    M      malignancy hazard accumulator monotone non-decreasing, in [0,2]
    7    flare  acute cholangitis flare      transient, decays

Context constants (supplied to a model, NOT predicted): disease_class, age, sex,
responder in {0,1}, udca_start (month), ercp_months (list).
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

# --- field indices (single source of truth; imported by models/eval) ---------------------
F, D, S, P, A, C, M, FLARE = range(8)
FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
N_FIELDS = 8

# fields that must never decrease month-to-month (S is handled separately: it may drop at ERCP)
MONOTONE_UP = (F, D, P, M)
FIELD_MAX = np.array([1, 1, 1, 1, 1, 1, 2, 1], dtype=np.float32)  # per-field upper bound


@dataclass
class Patient:
    """Context constants + hidden per-patient parameters that drive the dynamics."""
    # --- context: known to a model at prediction time ---
    disease_class: int          # 0..2, sets baseline inflammation/cholestasis "aggressiveness"
    age: float                  # normalised ~[0,1]
    sex: int                    # 0/1 (kept for realism; weak effect)
    responder: int              # 1 => treatment actually suppresses A and C
    udca_start: int             # month UDCA therapy begins (>= n_months => never)
    ercp_months: List[int] = field(default_factory=list)  # months an ERCP is performed
    # --- hidden: NOT given to a model, the thing generalisation must cope with ---
    susceptibility: float = 1.0  # per-patient multiplier on how fast the ratchets creep


def sample_patient(rng: np.random.Generator, n_months: int,
                   susc_range=None, udca_start_range=None) -> Patient:
    """Draw one patient's context + hidden parameters.

    susc_range=(lo,hi): rejection-sample susceptibility into a band. Used to CENSOR the
        training distribution (e.g. [0.5, 2.0]) and to build the held-out-susceptibility probe
        cohort (e.g. [2.0, 3.5]).
    udca_start_range=(lo,hi): force UDCA to begin in a fixed window. Used for the
        unseen-treatment-timing probe (e.g. late starts the model never trained on).
    """
    disease_class = int(rng.integers(0, 3))
    responder = int(rng.random() < 0.6)                      # 60% respond to therapy
    if udca_start_range is not None:
        udca_start = int(rng.integers(udca_start_range[0], udca_start_range[1]))
    else:
        # ~70% ever get UDCA, starting somewhere in the first ~half of the window
        udca_start = int(rng.integers(2, n_months // 2)) if rng.random() < 0.7 else n_months + 1
    # 0-2 ERCP events, only meaningful once strictures have had time to build
    n_ercp = int(rng.integers(0, 3))
    ercp_months = sorted(int(m) for m in rng.integers(6, n_months, size=n_ercp))
    susc = float(rng.lognormal(mean=0.0, sigma=0.80))        # median 1, wide spread
    if susc_range is not None:
        lo, hi = susc_range
        while not (lo <= susc <= hi):                        # reject until inside the band
            susc = float(rng.lognormal(mean=0.0, sigma=0.80))
    return Patient(
        disease_class=disease_class,
        age=float(rng.uniform(0.2, 0.9)),
        sex=int(rng.integers(0, 2)),
        responder=responder,
        udca_start=udca_start,
        ercp_months=ercp_months,
        susceptibility=susc,
    )


def simulate(p: Patient, n_months: int, rng: np.random.Generator) -> np.ndarray:
    """Roll one patient forward for n_months. Returns x of shape [n_months, 8]."""
    x = np.zeros((n_months, N_FIELDS), dtype=np.float32)

    # disease_class raises the "set point" of the fast, reversible channels A and C
    a_base = 0.15 + 0.10 * p.disease_class      # inflammatory set point
    c_base = 0.15 + 0.10 * p.disease_class      # cholestatic set point
    susc = p.susceptibility * (0.8 + 0.4 * p.age)  # older patients ratchet a little faster

    # initial state: everything low; A and C start near their baselines
    x[0, A] = np.clip(a_base + 0.05 * rng.standard_normal(), 0, 1)
    x[0, C] = np.clip(c_base + 0.05 * rng.standard_normal(), 0, 1)

    for t in range(1, n_months):
        prev = x[t - 1]
        cur = prev.copy()

        # treatment on/off this month: only responders actually get suppression
        on_udca = (t >= p.udca_start) and (p.responder == 1)
        supp = 0.6 if on_udca else 0.0          # 60% knock-down of A/C set points

        # --- flare (idx 7): transient. Random onset, then geometric decay. --------------
        # onset a bit more likely when strictures are high (cholangitis rides on obstruction)
        p_onset = 0.04 + 0.10 * prev[S]
        onset = 1.0 if rng.random() < p_onset else 0.0
        cur[FLARE] = max(onset, prev[FLARE] * 0.4)   # spike to 1 on onset, else decay x0.4

        # --- A (idx 4): fast mean-reverting toward a (possibly treated) set point + flare -
        a_set = a_base * (1 - supp)
        cur[A] = prev[A] + 0.5 * (a_set - prev[A]) + 0.5 * cur[FLARE] + 0.03 * rng.standard_normal()
        cur[A] = np.clip(cur[A], 0, 1)

        # --- C (idx 5): same shape as A; treatment suppresses it too ---------------------
        c_set = c_base * (1 - supp)
        cur[C] = prev[C] + 0.4 * (c_set - prev[C]) + 0.5 * cur[FLARE] + 0.03 * rng.standard_normal()
        cur[C] = np.clip(cur[C], 0, 1)

        # --- ratchets F, D, P (idx 0,1,3): non-negative creep driven by A and C ----------
        # increment >= 0 by construction (A,C,susc >= 0) => strictly non-decreasing.
        drive = susc * (0.6 * prev[A] + 0.4 * prev[C])
        cur[F] = min(prev[F] + 0.022 * drive, 1.0)
        cur[D] = min(prev[D] + 0.015 * drive, 1.0)
        cur[P] = min(prev[P] + 0.011 * (drive + 0.5 * prev[F]), 1.0)  # portal HTN also tracks fibrosis

        # --- S (idx 2): creeps up with inflammation; ERCP steps it DOWN (see D1) ---------
        cur[S] = min(prev[S] + 0.018 * susc * prev[A], 1.0)
        if t in p.ercp_months:
            cur[S] = max(cur[S] - 0.4, 0.0)          # transient mechanical relief

        # --- M (idx 6): hazard accumulator of sustained F*C. Monotone, capped at 2. ------
        cur[M] = min(prev[M] + 0.05 * prev[F] * prev[C], 2.0)

        x[t] = cur

    return x


def generate_dataset(n_patients: int, n_months: int, seed: int,
                     susc_range=None, udca_start_range=None):
    """Return (X, ctx): X is [n_patients, n_months, 8]; ctx is a list[Patient].

    susc_range / udca_start_range are passed to sample_patient to build censored training
    cohorts or out-of-distribution probe cohorts.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_patients, n_months, N_FIELDS), dtype=np.float32)
    ctx: List[Patient] = []
    for i in range(n_patients):
        p = sample_patient(rng, n_months, susc_range=susc_range, udca_start_range=udca_start_range)
        X[i] = simulate(p, n_months, rng)
        ctx.append(p)
    return X, ctx


if __name__ == "__main__":
    # quick self-check: dynamics + constraints hold in the generated data itself
    X, ctx = generate_dataset(n_patients=200, n_months=60, seed=0)
    dx = np.diff(X, axis=1)  # month-to-month changes, shape [N, T-1, 8]
    for idx in MONOTONE_UP:
        assert dx[:, :, idx].min() >= -1e-6, f"{FIELD_NAMES[idx]} decreased!"
    assert (X >= -1e-6).all() and (X <= FIELD_MAX + 1e-6).all(), "out of bounds!"
    print(f"OK  X={X.shape}  monotone fields never decrease; all fields in range.")
    print("per-field [min, mean, max] over dataset:")
    for i, name in enumerate(FIELD_NAMES):
        print(f"  {name:5s} [{X[...,i].min():.3f}, {X[...,i].mean():.3f}, {X[...,i].max():.3f}]")
