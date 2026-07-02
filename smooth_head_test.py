"""
EXPERIMENT (branch exp/distributional-head): test Codex's smooth bounded-increment head against
the shipped softplus+clamp head. Same hard guarantees, but no final clamp -> smooth gradients
everywhere (the reviewer's "dead gradient at the floor" concern).

Shipped head (models/constraints.py):  next = prev + softplus(raw);   clamp to [0, fmax]
Smooth head (this file):                next = prev + (fmax - prev) * sigmoid(raw)   in [prev, fmax]
  - ratchets F,D,P,M : bounded monotone increment, smooth, no clamp
  - free A,C,flare   : sigmoid (unchanged)
  - S                : creep = (1-S)*sigmoid  (== bounded increment with fmax_S=1), then
                       relief = is_ercp * s_pre * sigmoid(raw_relief)  -> stays in [0,1], smooth
  - M coupling       : increment gated by prev F*C (unchanged principle)

Question: does the smoother parameterisation match the shipped 0.033 ratchet MAE with 0 violations?
If yes, it is a strictly-nicer ("mathematically locked down") drop-in. If it costs accuracy, the
clamp form stays. Measure, don't assume.
"""
import numpy as np
import torch
import torch.nn as nn

from data import get_split, build_rollout_batch, T
from generator import N_FIELDS, F, D, S, P, M, C
from models.baseline import CTX_DIM
from eval import rollout_from, mae_over, RATCHETS

EPOCHS, PBATCH, LR, HIDDEN, L, K = 90, 16, 1e-3, 64, 6, 24
FREE_IDX = [4, 5, 7]


class SmoothHead(nn.Module):
    """Bounded-increment constraint head: guarantees WITHOUT a final clamp -> smooth gradients."""

    def __init__(self, couple_m=True):
        super().__init__()
        self.couple_m = couple_m
        self.register_buffer("fmax", torch.tensor([1, 1, 1, 1, 1, 1, 2, 1], dtype=torch.float32))
        free = torch.zeros(N_FIELDS, dtype=torch.bool); free[FREE_IDX] = True
        self.register_buffer("free_mask", free)
        s1 = torch.zeros(N_FIELDS); s1[S] = 1.0
        self.register_buffer("s_onehot", s1)
        m1 = torch.zeros(N_FIELDS); m1[M] = 1.0
        self.register_buffer("m_onehot", m1)

    def forward(self, raw_fields, raw_relief, prev_x, is_ercp):
        room = (self.fmax - prev_x).clamp(min=0.0)          # headroom to the ceiling
        inc = room * torch.sigmoid(raw_fields)              # in [0, room] -> next in [prev, fmax]
        if self.couple_m:
            fc = (prev_x[..., F] * prev_x[..., C]).unsqueeze(-1)
            inc = inc * (1.0 - self.m_onehot + self.m_onehot * fc)
        mono = prev_x + inc                                 # bounded monotone, smooth, NO clamp
        val = torch.sigmoid(raw_fields)                     # free fields: absolute value in (0,1)
        nxt = torch.where(self.free_mask, val, mono)
        s_pre = nxt[..., S]                                 # already prev_s + (1-prev_s)*sigmoid
        relief = is_ercp * s_pre * torch.sigmoid(raw_relief)   # in [0, s_pre] -> s stays >= 0
        nxt = nxt - relief.unsqueeze(-1) * self.s_onehot
        return nxt


class SmoothStep(nn.Module):
    def __init__(self, hidden=HIDDEN, couple_m=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FIELDS + CTX_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N_FIELDS + 1),
        )
        self.head = SmoothHead(couple_m=couple_m)

    def forward(self, x, ctx, is_ercp):
        raw = self.net(torch.cat([x, ctx], dim=-1))
        return self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], x, is_ercp)


def train(seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, _ = get_split()
    _, ctx, erc, X = build_rollout_batch(tr); n = X.shape[0]
    model = SmoothStep(); opt = torch.optim.Adam(model.parameters(), lr=LR); mse = nn.MSELoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, PBATCH):
            b = perm[i:i + PBATCH]; Xb, cb, eb = X[b], ctx[b], erc[b]
            loss = mse(model(Xb[:, :-1], cb[:, 1:], eb[:, 1:]), Xb[:, 1:])
            s = int(rng.integers(0, T - 1 - L)); cur = Xb[:, s]
            for k in range(L):
                cur = model(cur, cb[:, s + k + 1], eb[:, s + k + 1]); loss = loss + mse(cur, Xb[:, s + k + 1]) / L
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    model = train(); model.eval()
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    pred = rollout_from(model, Xv, cv, ev, K).numpy()
    acc = mae_over(pred, Xn, K + 1, T, RATCHETS)
    # violation check: any monotone field decrease off-ERCP, any out-of-bounds
    dif = np.diff(pred, axis=1)
    mono_idx = [F, D, P, M]
    dec = (dif[..., mono_idx] < -1e-6).sum()
    oob = ((pred < -1e-6) | (pred > np.array([1, 1, 1, 1, 1, 1, 2, 1]) + 1e-6)).sum()
    print(f"smooth-head ratchet MAE (K={K}) = {acc:.4f}   (shipped softplus+clamp head 0.033)")
    print(f"monotone decreases (off-diff, incl ERCP for S) = {int(dec)}   out-of-bounds = {int(oob)}")


if __name__ == "__main__":
    main()
