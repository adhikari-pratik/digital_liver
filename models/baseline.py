"""
Baseline world model: x(t) IS the latent (no learned representation).

A one-step predictor  x(t) -> x(t+1)  whose output head makes the hard constraints hold
*by construction*: the network's weights are free, yet the monotone fields can never
decrease and every field stays in bounds, no matter what the net outputs. Constraints are
a property of the parameterisation, not something the loss has to learn.

This is the honest peer the JEPA-style model must beat. It is also the simplest thing that
could satisfy the assignment's constraint requirement, so it de-risks the whole exercise.
"""

import numpy as np
import torch
import torch.nn as nn

from generator import (Patient, N_FIELDS, F, D, S, P, A, C, M, FLARE)
from models.constraints import ConstraintHead

# --- how each field is produced in the output head (see models/constraints.py) -----------
MONO_UP = [F, D, P, M]        # next = prev + softplus(raw)  => non-negative increment
FREE    = [A, C, FLARE]       # next = sigmoid(raw)          => bounded, may rise or fall
# S is handled on its own: creep up like a monotone field, minus a relief term at ERCP months

# context feature layout the model consumes at each step (see make_ctx):
#   disease_class one-hot (3) | age (1) | sex (1) | responder (1) | on_udca (1) | is_ercp (1)
CTX_DIM = 8


def make_ctx(p: Patient, T: int) -> np.ndarray:
    """Per-step context matrix [T, CTX_DIM] for one patient.

    on_udca and is_ercp are *time-varying* (derived from the treatment timeline), so the
    model can react to therapy starting and to ERCP events -- exactly the levers the
    generalisation probe will move to unseen values later.
    """
    ctx = np.zeros((T, CTX_DIM), dtype=np.float32)
    ctx[:, p.disease_class] = 1.0          # one-hot class in cols 0..2
    ctx[:, 3] = p.age
    ctx[:, 4] = p.sex
    ctx[:, 5] = p.responder
    for t in range(T):
        ctx[t, 6] = 1.0 if t >= p.udca_start else 0.0     # UDCA is on from its start month
        ctx[t, 7] = 1.0 if t in p.ercp_months else 0.0    # ERCP happens on specific months
    return ctx


class MonotoneStep(nn.Module):
    """One-step predictor with a constraint-enforcing output head."""

    def __init__(self, ctx_dim: int = CTX_DIM, hidden: int = 64, couple_m: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FIELDS + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N_FIELDS + 1),   # 8 field outputs + 1 extra "S relief" output
        )
        self.head = ConstraintHead(couple_m=couple_m)   # by-construction guarantee (+ optional M<-F*C coupling)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, is_ercp: torch.Tensor) -> torch.Tensor:
        """x:[...,8]  ctx:[...,CTX_DIM]  is_ercp:[...]  ->  next state [...,8]."""
        raw = self.net(torch.cat([x, ctx], dim=-1))
        return self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], x, is_ercp)

    @torch.no_grad()
    def rollout(self, x0, ctx_seq, is_ercp_seq):
        """Free-run from x0 for the full horizon, feeding predictions back in.

        x0:[B,8]  ctx_seq:[B,T,CTX_DIM]  is_ercp_seq:[B,T]  ->  preds [B,T,8] (preds[:,0]=x0).
        This is the honest test: errors compound because we never see ground truth after t0.
        """
        B, T, _ = ctx_seq.shape
        out = torch.empty(B, T, N_FIELDS, device=x0.device)
        out[:, 0] = x0
        cur = x0
        for t in range(1, T):
            cur = self.forward(cur, ctx_seq[:, t], is_ercp_seq[:, t])
            out[:, t] = cur
        return out
