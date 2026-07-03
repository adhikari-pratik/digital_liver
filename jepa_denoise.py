"""
Denoised-anchor TS-JEPA: the one principled shot at BEATING the baseline UNDER SENSOR NOISE.

Both the baseline and the standard TS-JEPA cumsum their forecast from the noisy observed anchor x[K],
so both inherit the anchor error -- and the baseline, being memoryless, structurally CANNOT denoise it
(it sees only that one noisy point). This variant adds a STATE-DECODE HEAD that maps the encoder's
embedding at month K to a DENOISED estimate x_hat[K] from the whole observed window, and anchors the
constraint-by-construction forecast on x_hat[K] instead of the raw noisy value. Monotone/bounded still
hold (cumsum of non-neg increments, clamped); the only thing given up is "passes exactly through the
noisy anchor" -- which under noise you do NOT want. Trained with sensor-noise augmentation so the state
head learns to denoise (online view noisy; target + reconstruction loss clean).

Built-in ABLATION: the SAME trained model is scored with the denoised anchor ON and OFF, isolating
whether window-denoising the anchor is really the mechanism. If denoised-anchor JEPA beats the baseline
under noise, it is a real advantage a memoryless model cannot have. If not, we report it honestly.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from data import get_split, build_rollout_batch
from eval import rollout_from, mae_over, RATCHETS
from models.baseline import MonotoneStep
from models.jepa import variance_loss, covariance_loss
from generator import FIELD_MAX, S, N_FIELDS
from ts_jepa import Encoder, obs_mask, D, RATCHET_UP, FAST, EPOCHS, PBATCH, LR, K_EVAL

T = 60
FMAX = torch.tensor(FIELD_MAX)
FMAXnp = np.array(FIELD_MAX, dtype=np.float32)
LAM_REC, LAM_INV, LAM_VAR, LAM_COV, LAM_STATE = 1.0, 1.0, 1.0, 1.0, 2.0
SIGMAS = [0.0, 0.05, 0.10, 0.15]
NOISE_TRAIN = 0.08
SEEDS = [0, 1, 2]                 # multi-seed gate: confirm the noise-axis win is seed-stable


class TSJepaDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = nn.Linear(D, N_FIELDS + 1)     # per-step increments + S-relief (as in ts_jepa)
        self.state_dec = nn.Linear(D, N_FIELDS)   # NEW: denoised current-state estimate from the window


def decode_denoised(m, z, X, is_ercp, K, denoised=True):
    """Constraint-by-construction decode, but anchor the cumsum on a DENOISED state estimate (from the
    encoder window) instead of the raw observed X[:,K] when denoised=True. Returns (forecast, state_est)."""
    raw = m.dec(z)
    inc = Fn.softplus(raw[..., :N_FIELDS])
    relief = Fn.softplus(raw[..., N_FIELDS]) * is_ercp
    state_est = torch.sigmoid(m.state_dec(z)) * FMAX.to(z.dtype)      # [B,T,N] denoised state estimate
    anchor = state_est[:, K] if denoised else X[:, K]                # [B,N]
    out = X.clone()
    fmax = FMAX.to(X.dtype)
    for i in RATCHET_UP:                                             # cumsum -> non-decreasing
        out[:, K + 1:, i] = (anchor[:, i:i + 1] + torch.cumsum(inc[:, K + 1:, i], 1)).clamp(0, fmax[i])
    s_step = inc[:, K + 1:, S] - relief[:, K + 1:]
    out[:, K + 1:, S] = (anchor[:, S:S + 1] + torch.cumsum(s_step, 1)).clamp(0, 1)
    for i in FAST:
        out[:, K + 1:, i] = torch.sigmoid(raw[:, K + 1:, i])
    return out, state_est


def train(seed=0, noise_std=NOISE_TRAIN):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    n = X.shape[0]
    m = TSJepaDenoise(); opt = torch.optim.Adam(m.parameters(), LR)
    tgt = copy.deepcopy(m.enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    allobs = torch.ones(PBATCH, T, dtype=torch.bool)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, PBATCH):
            b = perm[i:i + PBATCH]
            if len(b) < PBATCH:
                continue
            Xb, cb, eb = X[b], ctx[b], erc[b]
            K = int(rng.integers(8, 41))
            mask = obs_mask(PBATCH, K)
            Xin = torch.minimum((Xb + noise_std * torch.randn_like(Xb)).clamp(min=0.0), FMAX)  # noisy view
            zo = m.enc(Xin, cb, mask)
            with torch.no_grad():
                zt = tgt(Xb, cb, allobs)                                    # clean EMA target
            xhat, state_est = decode_denoised(m, zo, Xin, eb, K, denoised=True)   # forecast off denoised anchor
            xhat_a, _ = decode_denoised(m, zt.detach(), Xb, eb, K, denoised=False)  # dec-anchor (clean path)
            zf = zo[:, K + 1:].reshape(-1, D)
            loss = (LAM_REC * Fn.mse_loss(xhat[:, K + 1:], Xb[:, K + 1:])          # forecast clean future
                    + LAM_REC * Fn.mse_loss(xhat_a[:, K + 1:], Xb[:, K + 1:])      # dec-anchor
                    + LAM_STATE * Fn.mse_loss(state_est[:, :K + 1], Xb[:, :K + 1])  # DENOISE observed window
                    + LAM_INV * Fn.mse_loss(zo[:, K + 1:], zt[:, K + 1:].detach())
                    + LAM_VAR * variance_loss(zf) + LAM_COV * covariance_loss(zf))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for tp, sp in zip(tgt.parameters(), m.enc.parameters()):
                    tp.data.mul_(0.99).add_(sp.data, alpha=0.01)
        if (ep + 1) % 20 == 0:
            print(f"   [denoise seed {seed}] epoch {ep+1}/{EPOCHS}", flush=True)
    m.eval()
    return m


def load_baseline():
    ck = torch.load("checkpoints/baseline.pt")
    b = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    b.load_state_dict(ck["state_dict"]); b.eval()
    return b


def add_noise(Xn, sigma, K0, rng):
    Xc = Xn.copy()
    if sigma > 0:
        w = Xc[:, :K0 + 1]
        Xc[:, :K0 + 1] = np.clip(w + rng.normal(0, sigma, w.shape).astype(np.float32), 0.0, FMAXnp)
    return torch.tensor(Xc)


def forecast(m, X, c, e, K, denoised=True):
    with torch.no_grad():
        z = m.enc(X, c, obs_mask(X.shape[0], K))
        out, _ = decode_denoised(m, z, X, e, K, denoised)
    return out.numpy()


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    base = load_baseline()
    models = []
    for s in SEEDS:
        print(f"  training denoised-anchor TS-JEPA seed {s} (noise-aug)...", flush=True)
        models.append(train(seed=s))
    rng = np.random.default_rng(0)

    print(f"\n=== SENSOR NOISE at K0=24: ratchet MAE on [25..60] vs CLEAN truth "
          f"(TS-JEPA mean+/-sd over {len(SEEDS)} seeds; lower=better) ===")
    print(f"  {'sigma':>6} | {'baseline':>9} | {'JEPA denoise-anchor':>21} | {'JEPA raw-anchor(ablate)':>23} | best")
    for s in SIGMAS:
        Xc = add_noise(Xn, s, 24, rng)                       # one noised cohort per sigma, shared by all
        b = mae_over(rollout_from(base, Xc, cv, ev, 24).numpy(), Xn, 25, T, RATCHETS)
        jd = [mae_over(forecast(m, Xc, cv, ev, 24, denoised=True), Xn, 25, T, RATCHETS) for m in models]
        jr = [mae_over(forecast(m, Xc, cv, ev, 24, denoised=False), Xn, 25, T, RATCHETS) for m in models]
        jd_m, jd_s = float(np.mean(jd)), float(np.std(jd))
        jr_m, jr_s = float(np.mean(jr)), float(np.std(jr))
        best = min([("baseline", b), ("denoise", jd_m), ("raw", jr_m)], key=lambda t: t[1])[0]
        print(f"  {s:>6.2f} | {b:>9.4f} | {jd_m:>10.4f} +/- {jd_s:.4f} | "
              f"{jr_m:>12.4f} +/- {jr_s:.4f} | {best}")

    print("\n  denoise-anchor vs raw-anchor is the ABLATION (same model): denoise < raw as sigma rises")
    print("  => window-denoising the anchor is the mechanism. denoise < baseline => JEPA wins the noise")
    print("  axis with an advantage a memoryless model structurally cannot have. Reported either way.")


if __name__ == "__main__":
    main()
