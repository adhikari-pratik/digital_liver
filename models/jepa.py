"""
Minimal JEPA-style predictive model for the liver world model.

Difference from the baseline that matters: a GRU reads the trajectory-so-far and distills it
into a per-patient latent w. That is where the HIDDEN susceptibility can be inferred -- the one
thing the memoryless baseline structurally cannot recover. Everything else is kept small.

Pieces
  enc   : state encoder      x_t (8)                         -> z_t   (d_state)   [the JEPA embedding]
  gru   : patient encoder    [x ; ctx] over history          -> w_t   (d_patient) [history-aware]
  pred  : latent predictor   [z_t ; w ; ctx_{t+1}]           -> zhat_{t+1} (d_state)
  dec   : decoder            zhat_{t+1}                       -> raw fields (8) + S-relief (1)
  head  : ConstraintHead     (raw, prev_x, is_ercp)          -> x_hat_{t+1}  (same 0-violation guarantee)

Training signal (built in train_jepa.py) combines:
  - invariance : predicted latent matches the true future embedding  (JEPA core)
  - decode     : x_hat matches the true next state                   (accuracy + recon anchor)
  - VICReg     : variance + covariance terms on z                    (explicit anti-collapse)
Collapse metric: effective_rank(z) -- if it slides toward 1, the latent is collapsing.
"""

import torch
import torch.nn as nn

from generator import N_FIELDS
from models.baseline import CTX_DIM
from models.constraints import ConstraintHead


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class JEPA(nn.Module):
    def __init__(self, ctx_dim=CTX_DIM, d_state=16, d_patient=16, hidden=64):
        super().__init__()
        self.d_state = d_state
        self.enc = mlp(N_FIELDS, hidden, d_state)
        self.gru = nn.GRU(N_FIELDS + ctx_dim, d_patient, batch_first=True)
        self.pred = mlp(d_state + d_patient + ctx_dim, hidden, d_state)
        self.dec = nn.Linear(d_state, N_FIELDS + 1)
        self.rec = nn.Linear(d_state, N_FIELDS)      # reconstruction anchor: z_t -> x_t (absolute)
        self.head = ConstraintHead()

    def encode(self, x):
        return self.enc(x)

    def reconstruct(self, z):
        """Decode z_t back to the CURRENT state x_t (bounded). Forces z_t to retain full state
        info -> the real anti-collapse anchor. Without this the encoder collapses."""
        return torch.sigmoid(self.rec(z)) * self.head.fmax

    def patient_latent(self, x_seq, ctx_seq):
        """GRU hidden state at every month: H[:,t] summarises history 0..t. -> [B,T,d_patient]."""
        H, _ = self.gru(torch.cat([x_seq, ctx_seq], dim=-1))
        return H

    def step(self, z_t, w, ctx_tgt, prev_x, is_ercp):
        """One latent step + constrained decode. Returns (x_next, zhat)."""
        zhat = self.pred(torch.cat([z_t, w, ctx_tgt], dim=-1))
        raw = self.dec(zhat)
        x_next = self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], prev_x, is_ercp)
        return x_next, zhat

    @torch.no_grad()
    def rollout(self, X_true, ctx_seq, ercp_seq, K):
        """Infer w from the conditioning window 0..K, hold it fixed, free-roll K+1..H-1.

        Holding w fixed is the intended semantics: infer the patient's (hidden) progression
        speed from history, then predict forward with it. Returns full [B,H,8].
        """
        B, H, _ = X_true.shape
        w = self.patient_latent(X_true[:, :K + 1], ctx_seq[:, :K + 1])[:, K]   # [B,d_patient]
        out = X_true.clone()
        cur = X_true[:, K]
        for t in range(K + 1, H):
            z = self.encode(cur)
            cur, _ = self.step(z, w, ctx_seq[:, t], cur, ercp_seq[:, t])
            out[:, t] = cur
        return out


# --- anti-collapse: VICReg terms + the metric that would catch collapse -------------------

def variance_loss(z, gamma=1.0, eps=1e-4):
    """Hinge that pushes each latent dim's std up to >= gamma. Std->0 is collapse, so we
    penalise it directly. This is the main anti-collapse force."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(gamma - std).mean()


def covariance_loss(z):
    """Push OFF-diagonal covariances to 0 so dims don't all encode the same thing (a subtler
    collapse where the latent is high-variance but effectively 1-D)."""
    z = z - z.mean(dim=0)
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


@torch.no_grad()
def effective_rank(z, eps=1e-9):
    """Collapse METRIC: exp(entropy of normalised covariance eigenvalues). ~d_state = healthy
    (dims used evenly); ->1 means the latent has collapsed onto a single direction."""
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = ev / (ev.sum() + eps)
    return float(torch.exp(-(p * (p + eps).log()).sum()))
