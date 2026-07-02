"""
Boundary experiment (round 2, steelmanned): does the JEPA latent overtake raw-space prediction
once the stripped-out stochastic observation substrate is re-attached?

The main analysis (memo section 3, jepa_sweep.py) shows the latent is a NET COST on the clean 8-D
state. The brief itself says the real pipeline "has its own stochastic substrate on top of x(t)"
that was removed for this exercise. This script re-attaches it and asks the fair question: with the
substrate back, does JEPA win? We keep x(t) byte-for-byte identical -- only what the model SEES
changes.

Three fixes over the first failed attempt (DECISIONS.md D12):
  1. ENTANGLED nuisance -- the noise is spread across the SAME high-dim observation channels as the
     signal (an isotropic high-dim substrate), so a raw predictor cannot just zero-out "noise dims."
  2. FREE latent -- no by-construction increment head here (that head is what forces raw-space decode
     and kneecaps JEPA in the main task). This is the projection-style setup a skeptic would demand:
     let the latent be genuinely free and judge it on downstream signal recovery.
  3. JEPA gets its BEST anti-collapse shot -- we run both VICReg and an EMA/BYOL target and report
     both, instead of the VICReg variance term that backfired at high noise last time.

Fair scoring: every model is judged on recovering the TRUE CLEAN next state x(t+1) via a linear
probe on its predicted next-representation (raw -> predicted next observation; JEPA -> predicted
next latent). Capacity is small and matched, so wasting it on un-forecastable nuisance is a real
cost. Sweep the substrate strength sigma.

Honest expectation: raw-space prediction must carry the nuisance-dominated reconstruction target;
JEPA can abstract it. If a crossover appears, it validates the team's direction and maps WHEN it
pays. If it does not, we say so.

Run: python jepa_substrate.py
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_split, _ctx_matrix
from models.baseline import CTX_DIM
from generator import N_FIELDS

D_SIG = 8            # signal subspace dimension
D_OBS = 96           # observation dimension (signal entangled across all; nuisance fills all)
Z = 8               # small, matched latent bottleneck
HID = 48
EPOCHS = 40
SIGMAS = [0.0, 0.5, 1.0, 2.0, 4.0]   # nuisance strength, relative to unit-scaled signal


def mlp(i, o, h=HID):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = mlp(D_OBS, Z)
        self.pred = mlp(Z + CTX_DIM, Z)
        self.dec = mlp(Z, D_OBS)          # used only by the raw variant


def variance_loss(z, gamma=1.0, eps=1e-4):
    return torch.relu(gamma - torch.sqrt(z.var(0) + eps)).mean()


def covariance_loss(z):
    z = z - z.mean(0)
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / z.shape[1]


def make_obs(X_flat, Wsig, mu, sd, sigma, rng):
    """True state (rows of 8) -> observation. Signal = unit-scaled projection spread across all
    D_OBS dims (entangled); nuisance = isotropic noise on all dims (entangled with signal)."""
    sig = (X_flat @ Wsig.T - mu) / sd                                  # [N, D_OBS]
    noise = rng.normal(0, sigma, size=sig.shape).astype(np.float32)    # entangled: every dim
    return (sig + noise).astype(np.float32)


def train(kind, o_t, o_tp1, ctx, seed=0):
    """kind in {'raw','jepa_vicreg','jepa_ema'}."""
    torch.manual_seed(seed)
    m = Net()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    teacher = copy.deepcopy(m.enc)
    for p in teacher.parameters():
        p.requires_grad_(False)
    n = o_t.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, 2048):
            b = perm[i:i + 2048]
            z = m.enc(o_t[b])
            zhat = m.pred(torch.cat([z, ctx[b]], -1))
            if kind == "raw":
                loss = ((m.dec(zhat) - o_tp1[b]) ** 2).mean()
            elif kind == "jepa_vicreg":
                with torch.no_grad():
                    tgt = m.enc(o_tp1[b])
                loss = ((zhat - tgt) ** 2).mean() + variance_loss(z) + 0.5 * covariance_loss(z)
            else:  # jepa_ema  (BYOL-style target, no variance term to force noise in)
                with torch.no_grad():
                    tgt = teacher(o_tp1[b])
                loss = ((zhat - tgt) ** 2).mean() + 0.1 * covariance_loss(z)
            opt.zero_grad(); loss.backward(); opt.step()
            if kind == "jepa_ema":
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(), m.enc.parameters()):
                        tp.data.mul_(0.99).add_(sp.data, alpha=0.01)
    m.eval()
    return m


def feature(m, kind, o_t, ctx):
    with torch.no_grad():
        zhat = m.pred(torch.cat([m.enc(o_t), ctx], -1))
        return m.dec(zhat) if kind == "raw" else zhat


def probe(feat_tr, x_tr, feat_va):
    A = torch.cat([feat_tr, torch.ones(feat_tr.shape[0], 1)], 1)
    W = torch.linalg.lstsq(A, x_tr).solution
    return torch.cat([feat_va, torch.ones(feat_va.shape[0], 1)], 1) @ W


def pairs(X, ctxm):
    xt = X[:, :-1].reshape(-1, N_FIELDS)
    xtp1 = X[:, 1:].reshape(-1, N_FIELDS)
    ctx = ctxm[:, 1:].reshape(-1, ctxm.shape[-1])
    return xt, xtp1, ctx


def main():
    rng = np.random.default_rng(0)
    tr, va = get_split()
    Wsig = rng.normal(0, 1, size=(D_OBS, N_FIELDS)).astype(np.float32)   # entangled projection
    sig_tr = tr["X"].reshape(-1, N_FIELDS) @ Wsig.T
    mu, sd = sig_tr.mean(0), sig_tr.std(0) + 1e-6

    xt_tr, xtp1_tr, ctx_tr = pairs(tr["X"], _ctx_matrix(tr))
    xt_va, xtp1_va, ctx_va = pairs(va["X"], _ctx_matrix(va))
    ctxt_tr, ctxt_va = torch.tensor(ctx_tr), torch.tensor(ctx_va)
    x_tr_t, x_va_t = torch.tensor(xtp1_tr), torch.tensor(xtp1_va)

    kinds = ["raw", "jepa_vicreg", "jepa_ema"]
    print(f"obs dim {D_OBS} (8-D signal entangled across all dims + isotropic nuisance); "
          f"latent Z={Z}, hidden={HID}, {EPOCHS} epochs")
    print(f"scoring: linear probe of predicted next-representation -> TRUE clean x(t+1)\n")
    print(f"  {'sigma':>6} | {'raw':>8} | {'jepa(VICReg)':>13} | {'jepa(EMA)':>10} | best JEPA vs raw")
    print("  " + "-" * 66)
    results = {k: [] for k in kinds}
    for sigma in SIGMAS:
        ro = np.random.default_rng(100)
        o_t_tr = torch.tensor(make_obs(xt_tr, Wsig, mu, sd, sigma, ro))
        o_tp1_tr = torch.tensor(make_obs(xtp1_tr, Wsig, mu, sd, sigma, ro))
        o_t_va = torch.tensor(make_obs(xt_va, Wsig, mu, sd, sigma, ro))
        row = {}
        for k in kinds:
            m = train(k, o_t_tr, o_tp1_tr, ctxt_tr)
            f_tr = feature(m, k, o_t_tr, ctxt_tr)
            f_va = feature(m, k, o_t_va, ctxt_va)
            row[k] = float((probe(f_tr, x_tr_t, f_va) - x_va_t).abs().mean())
            results[k].append(row[k])
        best_j = min(row["jepa_vicreg"], row["jepa_ema"])
        verdict = f"JEPA wins by {row['raw'] - best_j:+.4f}" if best_j < row["raw"] else f"raw wins by {best_j - row['raw']:+.4f}"
        print(f"  {sigma:>6.1f} | {row['raw']:>8.4f} | {row['jepa_vicreg']:>13.4f} | {row['jepa_ema']:>10.4f} | {verdict}")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(SIGMAS, results["raw"], "o-", color="crimson", label="predict RAW observation")
    ax.plot(SIGMAS, results["jepa_vicreg"], "s-", color="steelblue", label="predict LATENT (JEPA, VICReg)")
    ax.plot(SIGMAS, results["jepa_ema"], "^-", color="seagreen", label="predict LATENT (JEPA, EMA/BYOL)")
    ax.set_xlabel("stochastic substrate strength  σ")
    ax.set_ylabel("MAE recovering true clean x(t+1)")
    ax.set_title("Re-attaching the stripped-out substrate: does the JEPA latent overtake raw?\n"
                 "(x(t) unchanged; only the observation is noisy/high-dim)")
    ax.legend(); ax.grid(alpha=0.3)
    import os
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/jepa_substrate.png", dpi=110)
    print("\nsaved figures/jepa_substrate.png")


if __name__ == "__main__":
    main()
