"""
Train the baseline MonotoneStep with a one-step + MULTI-STEP (short free-rollout) loss.

Upgrade over pure one-step training: rollout error is dominated by COMPOUNDING of tiny one-step
errors, so we also supervise a short free rollout from a random start each batch. This is the
single change that most improves rollout accuracy (measured: ~0.052 -> ~0.035 ratchet MAE at K=24).
At the end we print BOTH one-step and free-rollout error on held-out patients (the drift check).
"""

import numpy as np
import torch

from data import get_split, build_pairs, build_rollout_batch, T
from models.baseline import MonotoneStep, N_FIELDS
from eval import rollout_from, mae_over, RATCHETS
from generator import FIELD_NAMES

EPOCHS = 120
PBATCH = 16           # patients per batch (sequence training, for the multistep term)
LR = 1e-3
HIDDEN = 64
SEED = 0
L = 6                 # multistep horizon
COUPLE_M = True       # M<-F*C coupling by construction (the brief's "interesting part"): measured
                      # better than a freely-learned M (ratchet 0.0367->0.0325, M MAE ~halved), still
                      # 0-violation. Shipped by default (D20). Set False to reproduce the uncoupled peer.
CKPT = "checkpoints/baseline.pt"


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)     # [N,T,8], [N,T,CTX], [N,T]
    n = X.shape[0]
    print(f"train patients={n}  val patients={va['X'].shape[0]}  (one-step + {L}-step multistep)")

    model = MonotoneStep(hidden=HIDDEN, couple_m=COUPLE_M)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, PBATCH):
            b = perm[i:i + PBATCH]
            Xb, cb, eb = X[b], ctx[b], erc[b]
            loss = mse(model(Xb[:, :-1], cb[:, 1:], eb[:, 1:]), Xb[:, 1:])   # one-step, all months
            s = int(rng.integers(0, T - 1 - L))                              # short free rollout
            cur = Xb[:, s]
            for k in range(L):
                cur = model(cur, cb[:, s + k + 1], eb[:, s + k + 1])
                loss = loss + mse(cur, Xb[:, s + k + 1]) / L
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 40 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{EPOCHS}", flush=True)

    # --- report BOTH one-step and free-rollout error on val (drift check) -----------------
    model.eval()
    vx_in, vctx, verc, vx_tg = build_pairs(va)
    _, cv, ev, Xv = build_rollout_batch(va)
    with torch.no_grad():
        one_step = model(vx_in, vctx, verc).numpy()
        roll = rollout_from(model, Xv, cv, ev, 24).numpy()
    mae_1 = np.abs(one_step - vx_tg.numpy()).reshape(-1, N_FIELDS).mean(axis=0)
    ratchet_roll = mae_over(roll, Xv.numpy(), 25, T, RATCHETS)
    print("\nper-field one-step MAE (held-out):")
    print("  " + "  ".join(f"{FIELD_NAMES[i]}={mae_1[i]:.3f}" for i in range(N_FIELDS)))
    print(f"free-rollout ratchet MAE (K=24) = {ratchet_roll:.4f}   (target ~0.035)")

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": HIDDEN, "couple_m": COUPLE_M}, CKPT)
    print(f"saved {CKPT}")


if __name__ == "__main__":
    main()
