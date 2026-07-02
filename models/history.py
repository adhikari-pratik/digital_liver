"""
"Baseline + w": the disciplined model chosen after the JEPA dead-end (see DECISIONS.md D6).

It keeps the baseline's precise, native-space increment prediction (so constraints stay exact
and the tiny monotone increments keep their precision) and ADDS one thing: a GRU that reads the
trajectory-so-far into a per-patient latent `w`. `w` lets the model infer the HIDDEN
susceptibility from history -- the single benefit that actually measured -- without paying the
JEPA latent-bottleneck cost (decode error, collapse risk, constraint reconciliation).

Prediction is still  (x_t, ctx_{t+1}, w) -> constrained next state.  No latent-space rollout.
"""

import torch
import torch.nn as nn

from generator import N_FIELDS
from models.baseline import CTX_DIM
from models.constraints import ConstraintHead
from models.jepa import mlp


class HistoryStep(nn.Module):
    def __init__(self, ctx_dim: int = CTX_DIM, d_patient: int = 16, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(N_FIELDS + ctx_dim, d_patient, batch_first=True)
        self.net = mlp(N_FIELDS + ctx_dim + d_patient, hidden, N_FIELDS + 1)
        self.head = ConstraintHead()

    def patient_latent(self, x_seq, ctx_seq):
        """GRU hidden at every month; H[:,t] summarises history 0..t. -> [B,T,d_patient]."""
        H, _ = self.gru(torch.cat([x_seq, ctx_seq], dim=-1))
        return H

    def step(self, x_t, ctx_tgt, w, is_ercp):
        raw = self.net(torch.cat([x_t, ctx_tgt, w], dim=-1))
        return self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], x_t, is_ercp)

    @torch.no_grad()
    def rollout(self, X_true, ctx_seq, ercp_seq, K):
        """Infer w from window 0..K, hold fixed (static-susceptibility assumption, D5),
        then free-roll K+1..H-1 in native state space. Returns full [B,H,8]."""
        B, H, _ = X_true.shape
        w = self.patient_latent(X_true[:, :K + 1], ctx_seq[:, :K + 1])[:, K]
        out = X_true.clone()
        cur = X_true[:, K]
        for t in range(K + 1, H):
            cur = self.step(cur, ctx_seq[:, t], w, ercp_seq[:, t])
            out[:, t] = cur
        return out
