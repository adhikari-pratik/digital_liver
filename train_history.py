"""
Train "baseline + w" (HistoryStep): teacher-forced one-step, same protocol as the baseline.
Loss is just prediction MSE -- `w` is learned end-to-end through it (no VICReg needed, since
there is no latent-invariance objective that could collapse).
"""

import numpy as np
import torch

from data import get_split, build_rollout_batch
from models.history import HistoryStep
from generator import FIELD_NAMES, N_FIELDS

EPOCHS = 60
BATCH = 128
LR = 1e-3
SEED = 0
CKPT = "checkpoints/history.pt"


def main():
    torch.manual_seed(SEED)
    tr, va = get_split()
    _, ctx, ercp, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va)
    n = X.shape[0]

    model = HistoryStep()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            Xb, cb, eb = X[b], ctx[b], ercp[b]
            w = model.patient_latent(Xb, cb)[:, :-1]        # history up to t
            xhat = model.step(Xb[:, :-1], cb[:, 1:], w, eb[:, 1:])
            loss = mse(xhat, Xb[:, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  train MSE={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        roll = model.rollout(Xv, cv, ev, K=24).numpy()
    mae = np.abs(roll[:, 25:] - Xv.numpy()[:, 25:]).reshape(-1, N_FIELDS).mean(0)
    print("\nbaseline+w held-out free-rollout MAE (K=24):")
    print("  " + "  ".join(f"{FIELD_NAMES[i]}={mae[i]:.3f}" for i in range(N_FIELDS)))

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, CKPT)
    print(f"\nsaved {CKPT}")


if __name__ == "__main__":
    main()
