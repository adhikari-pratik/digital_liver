"""
!!! PRE-FIX EVIDENCE (see DECISIONS.md D14). These variants were run BEFORE the dec-anchor fix, so
their gap to the baseline reflects the same decoder/target-space bug, not the variants' true
ceiling. They are NOT current evidence for corrected JEPA behavior -- for that, see train_jepa.py
(dec-anchor GRU-JEPA, 0.12) and ts_jepa.py (masked TS-JEPA, 0.041). Kept as the round-2 record.

Round 2 on JEPA: honestly test whether better-tuned variants close the gap to the baseline
(K=24 ratchet MAE = 0.052), or whether the ceiling holds. We vary the three levers most likely
to matter, per the D6 diagnosis (latent bottleneck on a low-dim, near-deterministic state):

  - EMA/BYOL-style target  (modern JEPA anti-collapse instead of VICReg)
  - decode-weighted recipe (optimise prediction accuracy over latent-invariance)
  - larger latent          (relieve the bottleneck)

All else matched to the shipped protocol (teacher-forced one-step, 60 epochs, same data/seed).
Run: python jepa_variants.py
"""

import copy
import numpy as np
import torch

from data import get_split, build_rollout_batch
from models.jepa import JEPA, variance_loss, covariance_loss, effective_rank
from eval import mae_over, RATCHETS
from generator import N_FIELDS

EPOCHS, BATCH, LR, SEED, K = 60, 128, 1e-3, 0, 24

CONFIGS = [
    # name,                         d_state, decode, inv, var, cov, ema
    ("shipped recipe (ref)",             16,    1.0, 1.0, 1.0, 1.0, False),
    ("decode-weighted",                  16,   10.0, 0.1, 1.0, 1.0, False),
    ("EMA target (BYOL), no VICReg",     16,    1.0, 1.0, 0.0, 0.0, True),
    ("EMA + decode-weighted + big z=32",  32,   10.0, 1.0, 0.0, 0.0, True),
    ("decode-only (no invariance)",      16,   10.0, 0.0, 1.0, 1.0, False),
]


def train_variant(d_state, w_dec, w_inv, w_var, w_cov, ema):
    torch.manual_seed(SEED)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va)
    n = X.shape[0]
    m = JEPA(d_state=d_state)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    mse = torch.nn.MSELoss()
    teacher = copy.deepcopy(m.enc)                      # EMA target encoder (used only if ema)
    for p in teacher.parameters():
        p.requires_grad_(False)

    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            Xb, cb, eb = X[b], ctx[b], erc[b]
            H = m.patient_latent(Xb, cb)
            z = m.enc(Xb[:, :-1]); w = H[:, :-1]
            zhat = m.pred(torch.cat([z, w, cb[:, 1:]], dim=-1))
            raw = m.dec(zhat)
            xhat = m.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], Xb[:, :-1], eb[:, 1:])
            tgt = (teacher(Xb[:, 1:]) if ema else m.enc(Xb[:, 1:])).detach()
            recon = m.reconstruct(z)
            zf = z.reshape(-1, d_state)
            loss = (w_dec * mse(xhat, Xb[:, 1:]) + mse(recon, Xb[:, :-1])
                    + w_inv * mse(zhat, tgt) + w_var * variance_loss(zf) + w_cov * covariance_loss(zf))
            opt.zero_grad(); loss.backward(); opt.step()
            if ema:
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(), m.enc.parameters()):
                        tp.data.mul_(0.99).add_(sp.data, alpha=0.01)

    with torch.no_grad():
        roll = m.rollout(Xv, cv, ev, K).numpy()
        er = effective_rank(m.enc(Xv[:, :-1]).reshape(-1, d_state))
    ratchet = mae_over(roll, Xv.numpy(), K + 1, Xv.shape[1], RATCHETS)
    return ratchet, er


def main():
    print("!!! PRE-FIX (DECISIONS.md D14): run before the dec-anchor fix -> gap reflects that bug,")
    print("    not the variants' ceiling. Current JEPA evidence: train_jepa.py (0.12), ts_jepa.py (0.041).\n")
    print(f"baseline reference: K={K} ratchet MAE = 0.0522\n")
    print(f"  {'variant':36} {'ratchet MAE':>11}  {'eff_rank':>8}")
    for name, d, wd, wi, wv, wc, ema in CONFIGS:
        r, er = train_variant(d, wd, wi, wv, wc, ema)
        flag = "  <- beats baseline" if r < 0.0522 else ("  ~matches" if r < 0.065 else "")
        print(f"  {name:36} {r:11.4f}  {er:8.2f}{flag}")
    print("\n(baseline = 0.0522. 'matches' within ~25%. Interpretation printed by caller.)")


if __name__ == "__main__":
    main()
