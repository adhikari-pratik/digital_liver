"""
A genuine masked TS-JEPA (transformer, action-conditioned) -- the architecture we had NOT tested.
Settle 'does TS-JEPA beat our baseline' by measurement, not argument.

Design (faithful to masked-grid JEPA, adapted to forecasting + our constraints):
  - Transformer encoder over the (features x time) grid. To forecast, MASK the future months'
    STATE (replace with a mask token) but KEEP their known action/context tokens (on_udca, ERCP) --
    realistic: we know the treatment plan, not the future state. This is the masked-grid + action
    conditioning the writeup describes.
  - JEPA objective: predict the EMA target-encoder's embeddings at masked months (no value loss in
    the latent term) + VICReg anti-collapse.
  - Constraints by construction, parallel: decode per-step non-negative increments and CUMSUM from
    x[K] -> ratchets provably non-decreasing; S = up - ERCP-gated relief; fast fields = sigmoid.
  - dec-anchor (the fix we found): also decode the TARGET embeddings to values, so the decoder can
    decode the space the JEPA loss pulls toward.
Scored vs baseline on ratchet MAE (K=24). Baseline refs: 1-step ~0.047, +multistep ~0.035.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from data import get_split, build_rollout_batch, _ctx_matrix, get_probes, T_LONG
from models.baseline import CTX_DIM
from models.jepa import variance_loss, covariance_loss
from eval import mae_over, RATCHETS
from generator import N_FIELDS, FIELD_MAX, MONOTONE_UP, S

T = 60
D, LAYERS, HEADS, EPOCHS, PBATCH, LR, K_EVAL = 48, 2, 4, 60, 32, 1e-3, 24
# Explicit multi-term loss weights (were all implicitly 1.0). The four terms are:
#   rec  = dual-pathway reconstruction anchor  (decode BOTH online zhat AND EMA-target z -> x)
#   inv  = latent invariance (predict the target embedding; stop-grad on target)
#   var  = VICReg variance hinge (anti-collapse); cov = VICReg covariance (decorrelate)
LAM_REC, LAM_INV, LAM_VAR, LAM_COV = 1.0, 1.0, 1.0, 1.0
SCHEDULE = True    # rec-heavy early -> invariance later. Adopted (D21): equal-or-BETTER than flat on
                   # every measured axis (in-dist 0.0387 vs 0.0407, OOD held-out 0.092 vs 0.10, 0 viol)
                   # with no downside. The in-dist gap is within seed-noise, so we take it but do NOT
                   # over-claim it as a large improvement.
RATCHET_UP = list(MONOTONE_UP)       # F,D,P,M (pure non-decreasing)
FAST = [4, 5, 7]                     # A,C,flare (bounded, non-monotone)
FMAX = torch.tensor(FIELD_MAX)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_emb = nn.Linear(N_FIELDS, D)
        self.ctx_emb = nn.Linear(CTX_DIM, D)
        self.mask_tok = nn.Parameter(torch.randn(D) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, HEADS, 2 * D, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, LAYERS)

    def forward(self, X, ctx, obs):
        s = self.state_emb(X)
        s = torch.where(obs.unsqueeze(-1), s, self.mask_tok)   # masked months -> mask token
        return self.tr(s + self.ctx_emb(ctx) + self.pos[:, :X.shape[1]])


class TSJepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = nn.Linear(D, N_FIELDS + 1)


def decode_forecast(dec, z, X, is_ercp, K):
    """Parallel constraint-by-construction decode of embeddings z into a forecast for K+1..T."""
    raw = dec(z)
    inc = Fn.softplus(raw[..., :N_FIELDS])                       # per-step non-neg increments
    relief = Fn.softplus(raw[..., N_FIELDS]) * is_ercp           # ERCP-gated S relief
    out = X.clone()
    fmax = FMAX.to(X.dtype)
    for i in RATCHET_UP:                                         # cumsum -> non-decreasing
        out[:, K + 1:, i] = (X[:, K:K + 1, i] + torch.cumsum(inc[:, K + 1:, i], 1)).clamp(0, fmax[i])
    s_step = inc[:, K + 1:, S] - relief[:, K + 1:]
    out[:, K + 1:, S] = (X[:, K:K + 1, S] + torch.cumsum(s_step, 1)).clamp(0, 1)
    for i in FAST:
        out[:, K + 1:, i] = torch.sigmoid(raw[:, K + 1:, i])
    return out


def obs_mask(N, K):
    m = torch.zeros(N, T, dtype=torch.bool); m[:, :K + 1] = True
    return m


def eval_ratchet(m, batch):
    """Ratchet MAE (K=24) on a prebuilt (_, ctx, ercp, X) cohort. Returns None if the cohort's
    horizon exceeds T: this masked transformer uses LEARNED ABSOLUTE positions, so it cannot
    address months beyond its trained horizon (unlike the recurrent baseline, which extrapolates
    natively). We surface that as an honest architectural limitation rather than fake a number."""
    _, c, e, X = batch
    H = X.shape[1]
    if H > T:
        return None
    with torch.no_grad():
        xh = decode_forecast(m.dec, m.enc(X, c, obs_mask(X.shape[0], K_EVAL)), X, e, K_EVAL).numpy()
    return mae_over(xh, X.numpy(), K_EVAL + 1, H, RATCHETS)


def train(seed=0, probe_batches=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va)
    n = X.shape[0]
    m = TSJepa(); opt = torch.optim.Adam(m.parameters(), LR)
    tgt = copy.deepcopy(m.enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    allobs = torch.ones(PBATCH, T, dtype=torch.bool)

    for ep in range(EPOCHS):
        if SCHEDULE:                                   # rec-heavy early -> invariance-leaning later
            f = ep / EPOCHS
            w_rec = LAM_REC * (1.0 + 2.0 * (1.0 - f))  # 3.0 -> 1.0
            w_inv = LAM_INV * (0.5 + 0.5 * f)          # 0.5 -> 1.0
        else:
            w_rec, w_inv = LAM_REC, LAM_INV            # flat (proven default)
        perm = torch.randperm(n)
        for i in range(0, n, PBATCH):
            b = perm[i:i + PBATCH]
            if len(b) < PBATCH:
                continue
            Xb, cb, eb = X[b], ctx[b], erc[b]
            K = int(rng.integers(8, 41))
            zo = m.enc(Xb, cb, obs_mask(PBATCH, K))              # online (masked-future)
            with torch.no_grad():
                zt = tgt(Xb, cb, allobs)                         # EMA target (full)
            xhat = decode_forecast(m.dec, zo, Xb, eb, K)
            xhat_a = decode_forecast(m.dec, zt.detach(), Xb, eb, K)   # dec-anchor
            zf = zo[:, K + 1:].reshape(-1, D)
            loss = (w_rec * Fn.mse_loss(xhat[:, K + 1:], Xb[:, K + 1:])       # rec: online value forecast
                    + w_rec * Fn.mse_loss(xhat_a[:, K + 1:], Xb[:, K + 1:])   # rec: dec-anchor (target path)
                    + w_inv * Fn.mse_loss(zo[:, K + 1:], zt[:, K + 1:].detach())  # inv: JEPA latent
                    + LAM_VAR * variance_loss(zf) + LAM_COV * covariance_loss(zf))  # var + cov (anti-collapse)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for tp, sp in zip(tgt.parameters(), m.enc.parameters()):
                    tp.data.mul_(0.99).add_(sp.data, alpha=0.01)
        if (ep + 1) % 20 == 0:
            print(f"   [seed {seed}] epoch {ep+1}/{EPOCHS}", flush=True)

    m.eval()
    with torch.no_grad():
        nv = Xv.shape[0]
        xh = decode_forecast(m.dec, m.enc(Xv, cv, obs_mask(nv, K_EVAL)), Xv, ev, K_EVAL).numpy()
    # constraint check
    d = np.diff(xh[:, K_EVAL:], axis=1)
    viol = sum(int((d[:, :, i] < -1e-6).sum()) for i in RATCHET_UP)
    ood = {name: eval_ratchet(m, b) for name, b in (probe_batches or {}).items()}
    return mae_over(xh, Xv.numpy(), K_EVAL + 1, T, RATCHETS), viol, ood


# baseline (coupled, D20) OOD ratchet MAE (K=24) from eval.py, for side-by-side context
BASE_REF = {"held-out susceptibility": "0.099 (3x)",
            "unseen treatment timing": "0.031 (unchanged)",
            "longer-than-training":    "0.100 (baseline extrapolates natively)"}


def main():
    print("TS-JEPA (masked transformer, action-conditioned). Baseline refs: 1-step 0.047, +MS+coupled 0.033\n")
    probe_batches = {name: build_rollout_batch(sp) for name, sp in get_probes(n=200).items()}
    maes, oods = [], []
    for sd in (0, 1, 2, 3, 4):
        mae, viol, ood = train(sd, probe_batches)
        maes.append(mae); oods.append(ood)
        print(f"  seed {sd}: ratchet MAE (K=24) = {mae:.4f}   ratchet violations = {viol}", flush=True)
    a = np.array(maes)
    print(f"\n  TS-JEPA over {len(a)} seeds: mean={a.mean():.4f} std={a.std():.4f} "
          f"min={a.min():.4f} max={a.max():.4f}")
    print("  baseline: +multistep+coupled 0.033 (std ~0.001), 1-step 0.047.")

    # --- OOD generalisation probe: same cohorts as eval.py/compare.py, averaged over seeds -------
    print("\n  === OOD generalisation probe (ratchet MAE K=24, TS-JEPA mean over seeds vs baseline) ===")
    print(f"  {'cohort':26s} | {'TS-JEPA':>8s} | baseline")
    for name in probe_batches:
        vals = [o[name] for o in oods if o.get(name) is not None]
        if vals:
            print(f"  {name:26s} | {np.mean(vals):>8.4f} | {BASE_REF[name]}")
        else:
            print(f"  {name:26s} | {'n/a':>8s} | {BASE_REF[name]}")
    print(f"  NOTE: 'longer-than-training' is n/a for TS-JEPA -- learned ABSOLUTE positions cap it at")
    print(f"  the trained horizon (T={T}); it cannot roll to {T_LONG} months without positional")
    print(f"  extrapolation. The recurrent baseline has no such cap -- a real architectural tradeoff.")


if __name__ == "__main__":
    main()
