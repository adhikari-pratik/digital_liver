"""
Generate REAL training + validation curves from actual training runs (nothing hand-drawn): for each
model, log the metric on a TRAIN subset and on the held-out VAL set every epoch, save the raw
per-epoch arrays (so the plot is verifiable) and a figure. The train-vs-val gap is the honest
generalisation signal; the plateau shows the epoch budget is sufficient, not cut short.

Outputs:
  figures/training_curve_baseline.png  + training_curves_baseline.npz
  figures/training_curve_tsjepa.png    + training_curves_tsjepa.npz
Run: python make_training_curves.py   (a few minutes on CPU)
"""
import copy
import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_split, build_rollout_batch, T
from models.baseline import MonotoneStep
from eval import rollout_from, mae_over, RATCHETS
import ts_jepa
from ts_jepa import TSJepa, decode_forecast, obs_mask, D as TSD, RATCHET_UP

K = 24


def ratchet_mae(model_roll, true):
    return mae_over(model_roll, true, K + 1, T, RATCHETS)


def baseline_curve():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va)
    # a fixed TRAIN subset (same size as val) for an apples-to-apples train-vs-val curve
    ntr = Xv.shape[0]
    Xs, cs, es = X[:ntr], ctx[:ntr], erc[:ntr]
    n = X.shape[0]
    m = MonotoneStep(hidden=64, couple_m=True)
    opt = torch.optim.Adam(m.parameters(), 1e-3); mse = torch.nn.MSELoss()
    EPOCHS, PB, L = 120, 16, 6
    ep_ax, tr_mae, va_mae = [], [], []

    def ev_mae():
        m.eval()
        with torch.no_grad():
            rtr = rollout_from(m, Xs, cs, es, K).numpy()
            rva = rollout_from(m, Xv, cv, ev, K).numpy()
        return ratchet_mae(rtr, Xs.numpy()), ratchet_mae(rva, Xv.numpy())

    for ep in range(1, EPOCHS + 1):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, PB):
            b = perm[i:i + PB]; Xb, cb, eb = X[b], ctx[b], erc[b]
            loss = mse(m(Xb[:, :-1], cb[:, 1:], eb[:, 1:]), Xb[:, 1:])
            s = int(rng.integers(0, T - 1 - L)); cur = Xb[:, s]
            for k in range(L):
                cur = m(cur, cb[:, s + k + 1], eb[:, s + k + 1]); loss = loss + mse(cur, Xb[:, s + k + 1]) / L
            opt.zero_grad(); loss.backward(); opt.step()
        if ep == 1 or ep % 5 == 0:
            t_, v_ = ev_mae(); ep_ax.append(ep); tr_mae.append(t_); va_mae.append(v_)
            print(f"[baseline] epoch {ep:3d}: train MAE={t_:.4f}  val MAE={v_:.4f}", flush=True)
    return np.array(ep_ax), np.array(tr_mae), np.array(va_mae)


def tsjepa_curve(seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va)
    ntr = Xv.shape[0]; Xs, cs, es = X[:ntr], ctx[:ntr], erc[:ntr]
    n = X.shape[0]
    m = TSJepa(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    tgt = copy.deepcopy(m.enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    PB, EPOCHS = 32, 60
    allobs = torch.ones(PB, T, dtype=torch.bool)
    ep_ax, tr_mae, va_mae = [], [], []

    def ev_mae(Xe, ce, ee):
        with torch.no_grad():
            xh = decode_forecast(m.dec, m.enc(Xe, ce, obs_mask(Xe.shape[0], K)), Xe, ee, K).numpy()
        return ratchet_mae(xh, Xe.numpy())

    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        for i in range(0, n, PB):
            b = perm[i:i + PB]
            if len(b) < PB:
                continue
            Xb, cb, eb = X[b], ctx[b], erc[b]
            Kk = int(rng.integers(8, 41))
            zo = m.enc(Xb, cb, obs_mask(PB, Kk))
            with torch.no_grad():
                zt = tgt(Xb, cb, allobs)
            xhat = decode_forecast(m.dec, zo, Xb, eb, Kk)
            xhat_a = decode_forecast(m.dec, zt.detach(), Xb, eb, Kk)
            zf = zo[:, Kk + 1:].reshape(-1, TSD)
            loss = (Fn.mse_loss(xhat[:, Kk + 1:], Xb[:, Kk + 1:]) + Fn.mse_loss(xhat_a[:, Kk + 1:], Xb[:, Kk + 1:])
                    + Fn.mse_loss(zo[:, Kk + 1:], zt[:, Kk + 1:].detach())
                    + ts_jepa.variance_loss(zf) + ts_jepa.covariance_loss(zf))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for tp, sp in zip(tgt.parameters(), m.enc.parameters()):
                    tp.data.mul_(0.99).add_(sp.data, alpha=0.01)
        if ep == 1 or ep % 5 == 0:
            m.eval(); t_ = ev_mae(Xs, cs, es); v_ = ev_mae(Xv, cv, ev)
            ep_ax.append(ep); tr_mae.append(t_); va_mae.append(v_)
            print(f"[ts-jepa]  epoch {ep:3d}: train MAE={t_:.4f}  val MAE={v_:.4f}", flush=True)
    return np.array(ep_ax), np.array(tr_mae), np.array(va_mae)


def save_curve(name, ep, trm, vam, final_ref):
    np.savez(f"training_curves_{name}.npz", epoch=ep, train_mae=trm, val_mae=vam)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ep, trm, "o-", color="steelblue", label="train ratchet MAE", ms=3)
    ax.plot(ep, vam, "s-", color="crimson", label="held-out (val) ratchet MAE", ms=3)
    ax.axhline(final_ref, ls="--", color="gray", lw=1, label=f"shipped ref ({final_ref})")
    ax.set_xlabel("epoch"); ax.set_ylabel("ratchet MAE (K=24, free rollout)")
    ax.set_title(f"{name}: real training vs validation curve (actual run, seed 0)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(f"figures/training_curve_{name}.png", dpi=130); plt.close(fig)
    print(f"  saved figures/training_curve_{name}.png  (+ training_curves_{name}.npz)")


def main():
    import os; os.makedirs("figures", exist_ok=True)
    print("=== baseline (multistep, coupled) ===")
    ep, trm, vam = baseline_curve(); save_curve("baseline", ep, trm, vam, 0.033)
    print("\n=== TS-JEPA (masked transformer, seed 0) ===")
    ep, trm, vam = tsjepa_curve(); save_curve("tsjepa", ep, trm, vam, 0.041)


if __name__ == "__main__":
    main()
