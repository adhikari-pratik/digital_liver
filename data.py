"""
Shared data plumbing so train.py and eval.py agree on EXACTLY the same held-out patients
and the same input/target alignment. If these diverged, "held-out accuracy" would be a lie.

Alignment convention (important):
  to predict x[t+1] we feed the state x[t] together with the CONTEXT OF THE TARGET MONTH t+1
  (on_udca and is_ercp at t+1). This matches the generator, which applies UDCA suppression and
  ERCP relief in the month they occur.
"""

import numpy as np
import torch

from generator import generate_dataset
from models.baseline import make_ctx, CTX_DIM

# --- dataset config (single source of truth) ---------------------------------------------
N_PATIENTS = 1000
T = 60
SEED = 0
VAL_FRAC = 0.2

# The model trains ONLY on this susceptibility band and this treatment-timing regime.
# The generalisation probes live deliberately OUTSIDE these, so "held-out" is truthful.
TRAIN_SUSC = (0.5, 2.0)          # in-distribution susceptibility band
PROBE_SUSC = (2.0, 3.5)          # unseen faster progressors
PROBE_UDCA = (35, 50)            # unseen LATE treatment starts (training uses months 2..29)
T_LONG = 96                      # longer-than-training horizon probe (train horizon is 60)


def get_split():
    """Deterministic train/val split by patient (in-distribution)."""
    X, ctx = generate_dataset(N_PATIENTS, T, seed=SEED, susc_range=TRAIN_SUSC)
    n_val = int(N_PATIENTS * VAL_FRAC)
    tr = {"X": X[:-n_val], "ctx": ctx[:-n_val]}   # first (1-VAL_FRAC) patients
    va = {"X": X[-n_val:], "ctx": ctx[-n_val:]}   # last VAL_FRAC patients, never trained on
    return tr, va


def get_probes(n=200):
    """Out-of-distribution cohorts for the generalisation probe. Each is a split dict."""
    hs_X, hs_c = generate_dataset(n, T, seed=SEED + 1, susc_range=PROBE_SUSC)
    lt_X, lt_c = generate_dataset(n, T, seed=SEED + 2, susc_range=TRAIN_SUSC,
                                  udca_start_range=PROBE_UDCA)
    lh_X, lh_c = generate_dataset(n, T_LONG, seed=SEED + 3, susc_range=TRAIN_SUSC)
    return {
        "held-out susceptibility": {"X": hs_X, "ctx": hs_c},
        "unseen treatment timing": {"X": lt_X, "ctx": lt_c},
        "longer-than-training":    {"X": lh_X, "ctx": lh_c},   # horizon T_LONG, not T
    }


def _ctx_matrix(split):
    """[N, H, CTX_DIM] context for every patient/month; H = the split's own horizon."""
    H = split["X"].shape[1]
    return np.stack([make_ctx(p, H) for p in split["ctx"]]).astype(np.float32)


def build_pairs(split):
    """Flattened one-step (teacher-forced) training pairs.

    Returns tensors: x_in [P,8], ctx_tgt [P,CTX_DIM], ercp_tgt [P], x_tgt [P,8]
    where P = N * (T-1).
    """
    X = split["X"]                       # [N,T,8]
    ctxm = _ctx_matrix(split)            # [N,T,CTX_DIM]
    x_in   = X[:, :-1]                   # state at months 0..T-2
    ctx_tg = ctxm[:, 1:]                 # context at target months 1..T-1
    x_tg   = X[:, 1:]                    # target state at months 1..T-1
    ercp   = ctx_tg[..., 7]             # is_ercp of the target month
    flat = lambda a: torch.tensor(a.reshape(-1, a.shape[-1]))
    return flat(x_in), flat(ctx_tg), torch.tensor(ercp.reshape(-1)), flat(x_tg)


def build_rollout_batch(split):
    """Whole-trajectory tensors for free-run rollout eval.

    Returns: x0 [N,8], ctx_seq [N,T,CTX_DIM], ercp_seq [N,T], X_true [N,T,8].
    """
    X = split["X"]
    ctxm = _ctx_matrix(split)
    x0 = torch.tensor(X[:, 0])
    return x0, torch.tensor(ctxm), torch.tensor(ctxm[..., 7]), torch.tensor(X)
