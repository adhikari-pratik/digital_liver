"""
Train the JEPA-style model (teacher-forced one-step, matching the baseline's protocol so the
head-to-head is fair).

Loss = decode_MSE + inv * latent_invariance + var*variance + cov*covariance
  decode  : x_hat vs true next state   (accuracy + reconstruction anchor)
  inv     : predicted latent vs true future embedding (stop-grad target)  -- the JEPA core
  var,cov : VICReg anti-collapse terms

We log effective_rank every epoch and FLAG if it slides toward 1 despite the variance term.
We also train a no-VICReg ablation to show what the anti-collapse terms actually buy.
"""

import numpy as np
import torch

from data import get_split, build_rollout_batch
from models.jepa import JEPA, variance_loss, covariance_loss, effective_rank
from generator import FIELD_NAMES, N_FIELDS

EPOCHS = 60
BATCH = 128          # patients per batch (we train on whole sequences for the GRU)
LR = 1e-3
LAM_INV, LAM_VAR, LAM_COV = 1.0, 1.0, 1.0
SEED = 0
CKPT = "checkpoints/jepa.pt"


def seq_tensors(split):
    _, ctx_seq, ercp_seq, X_true = build_rollout_batch(split)
    return X_true, ctx_seq, ercp_seq


def train_one(use_vicreg: bool, verbose: bool):
    torch.manual_seed(SEED)
    tr, _ = get_split()
    X, ctx, ercp = seq_tensors(tr)
    n = X.shape[0]
    model = JEPA()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = torch.nn.MSELoss()
    ranks = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            Xb, ctxb, ercpb = X[b], ctx[b], ercp[b]
            H = model.patient_latent(Xb, ctxb)          # [B,T,dp]
            z = model.enc(Xb[:, :-1])                   # [B,T-1,ds]
            w = H[:, :-1]                               # history up to t
            zhat = model.pred(torch.cat([z, w, ctxb[:, 1:]], dim=-1))
            raw = model.dec(zhat)
            xhat = model.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], Xb[:, :-1], ercpb[:, 1:])
            target = model.enc(Xb[:, 1:])               # future embedding (invariance target)
            zf = z.reshape(-1, model.d_state)
            recon = model.reconstruct(z)                 # z_t -> x_t (anti-collapse anchor)

            # dec-anchor (D14, the fix): the invariance loss drags zhat toward enc(x_{t+1}), but
            # `dec` is only ever trained on the PREDICTED zhat -- it never learns to decode the
            # target space. So decode the true embedding through the SAME head and match it to
            # x_{t+1}. This one term cut the JEPA gap from 0.52 -> 0.12 (see DECISIONS.md D14).
            raw_t = model.dec(target)
            xhat_t = model.head(raw_t[..., :N_FIELDS], raw_t[..., N_FIELDS], Xb[:, :-1], ercpb[:, 1:])

            loss = (mse(xhat, Xb[:, 1:]) + mse(recon, Xb[:, :-1])
                    + LAM_INV * mse(zhat, target.detach()) + mse(xhat_t, Xb[:, 1:]))
            if use_vicreg:
                loss = loss + LAM_VAR * variance_loss(zf) + LAM_COV * covariance_loss(zf)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            zf = model.enc(X[:, :-1]).reshape(-1, model.d_state)
            er = effective_rank(zf)
        ranks.append(er)
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  epoch {ep:3d}  loss={loss.item():.4f}  eff_rank(z)={er:.2f} / {model.d_state}")
    return model, ranks


def main():
    print("=== training JEPA (with VICReg) ===")
    model, ranks = train_one(use_vicreg=True, verbose=True)
    # Judge the CONVERGED rank against the state's intrinsic dim (~2.83), not the nominal 16 and
    # not the early transient -- a healthy latent here sits near ~3 (see DECISIONS.md D6).
    final_er = sum(ranks[-5:]) / 5
    print(f"effective rank: start {ranks[0]:.2f} -> converged {final_er:.2f}  (transient min {min(ranks):.2f})")
    if final_er < 2.0:                      # well below intrinsic dim ~2.8 => real collapse
        print("  !!! FLAG: converged effective rank < 2 -- latent is collapsing despite VICReg.")
    else:
        print(f"  OK: converged rank {final_er:.2f} is healthy vs intrinsic dim ~2.8 (not a collapse).")

    print("\n=== ablation: JEPA WITHOUT VICReg (anti-collapse off) ===")
    _, ranks_ab = train_one(use_vicreg=False, verbose=False)
    print(f"effective rank: start {ranks_ab[0]:.2f} -> end {ranks_ab[-1]:.2f}  (min {min(ranks_ab):.2f})")
    print(f"  => VICReg keeps eff_rank at {ranks[-1]:.2f}; without it, {ranks_ab[-1]:.2f}. "
          f"{'Collapse averted.' if ranks[-1] > ranks_ab[-1] + 0.5 else 'Little difference (see notes).'}")

    # quick val accuracy for the with-VICReg model
    _, va = get_split()
    Xv, ctxv, ercpv = seq_tensors(va)
    with torch.no_grad():
        roll = model.rollout(Xv, ctxv, ercpv, K=24).numpy()
    mae = np.abs(roll[:, 25:] - Xv.numpy()[:, 25:]).reshape(-1, N_FIELDS).mean(0)
    print("\nJEPA held-out free-rollout MAE (K=24):")
    print("  " + "  ".join(f"{FIELD_NAMES[i]}={mae[i]:.3f}" for i in range(N_FIELDS)))

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, CKPT)
    print(f"\nsaved {CKPT}")


if __name__ == "__main__":
    main()
