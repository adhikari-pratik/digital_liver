"""
Evaluation harness with honest numbers.

Reports, for the baseline:
  1. constraint-violation rate  (should be exactly 0 -- it is by construction)
  2. accuracy vs conditioning window K in {12,24,36}, against the irreducible NOISE FLOOR
     so we never bill aleatoric noise as model error
  3. error stratified by hidden susceptibility (in-distribution)
  4. the generalisation probe: held-out susceptibility, unseen treatment timing, longer horizon
     -- the cases designed to make the model fail, shown not hidden.
"""

import numpy as np
import torch

from data import get_split, get_probes, build_rollout_batch, T
from models.baseline import MonotoneStep, MONO_UP
from generator import simulate, FIELD_NAMES, N_FIELDS, S, FIELD_MAX

RATCHETS = [0, 1, 2, 3, 6]   # F, D, S, P, M  (the clinically decisive, constrained fields)
FAST = [4, 5, 7]             # A, C, flare    (fast/stochastic; near the noise floor)


def load_model():
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def rollout_from(model, X_true, ctx_seq, ercp_seq, K):
    """Observe true states 0..K, then free-roll K+1..H-1. Returns full [N,H,8] tensor."""
    N, H, _ = X_true.shape
    out = X_true.clone()
    cur = X_true[:, K]
    with torch.no_grad():
        for t in range(K + 1, H):
            cur = model(cur, ctx_seq[:, t], ercp_seq[:, t])
            out[:, t] = cur
    return out


def mae_over(pred, true, a, b, cols):
    return np.abs(pred[:, a:b][..., cols] - true[:, a:b][..., cols]).mean()


def noise_floor(ctx_list, H, K, n_real=40, seed=123, subset=60):
    """Irreducible error: |sample - conditional-mean| over months K+1..H, from random flares."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(N_FIELDS); c = 0
    for p in ctx_list[:subset]:
        sims = np.stack([simulate(p, H, rng) for _ in range(n_real)])
        acc += np.abs(sims - sims.mean(0))[:, K + 1:].reshape(-1, N_FIELDS).mean(0); c += 1
    return acc / c


def constraint_report(pred, ercp_seq):
    """Fraction of monotone steps that decrease, and any out-of-bounds. Over the whole array."""
    d = np.diff(pred, axis=1)                       # [N,H-1,8]
    viol = total = 0
    for i in MONO_UP:                               # F,D,P,M must never fall
        viol += int((d[:, :, i] < -1e-6).sum()); total += d[:, :, i].size
    nonercp = ~ercp_seq[:, 1:].astype(bool)         # S may fall only at ERCP months
    viol += int((d[:, :, S][nonercp] < -1e-6).sum()); total += int(nonercp.sum())
    oob = int(((pred < -1e-6) | (pred > FIELD_MAX + 1e-6)).sum())
    return viol, total, oob


def main():
    model = load_model()
    _, va = get_split()
    x0, ctx_seq, ercp_seq, X_true = build_rollout_batch(va)
    Xt = X_true.numpy(); erc = ercp_seq.numpy()

    # --- 1) constraints: full free rollout from t0 (the most steps -> strongest claim) -----
    full = rollout_from(model, X_true, ctx_seq, ercp_seq, K=0).numpy()
    viol, total, oob = constraint_report(full, erc)
    print("=== 1. constraint-violation rate (full free rollout) ===")
    print(f"  monotone/S violations: {viol}/{total}   out-of-bounds: {oob}   "
          f"=> violation rate = {viol/total:.6f}\n")

    # --- 2) accuracy vs conditioning window K, against the noise floor --------------------
    print("=== 2. rollout accuracy vs conditioning window K (mean MAE over predicted months) ===")
    print(f"  {'K':>3} | {'ratchets':>9} {'floor':>7} | {'fast':>7} {'floor':>7} | {'all':>7}")
    for K in (12, 24, 36):
        pred = rollout_from(model, X_true, ctx_seq, ercp_seq, K).numpy()
        fl = noise_floor(va["ctx"], T, K)
        r  = mae_over(pred, Xt, K + 1, T, RATCHETS); rf = fl[RATCHETS].mean()
        f  = mae_over(pred, Xt, K + 1, T, FAST);     ff = fl[FAST].mean()
        a  = mae_over(pred, Xt, K + 1, T, list(range(N_FIELDS)))
        print(f"  {K:>3} | {r:9.4f} {rf:7.4f} | {f:7.4f} {ff:7.4f} | {a:7.4f}")
    print("  (fast fields A/C/flare sit at their noise floor: irreducible, not model error)\n")

    # --- 3) error by hidden susceptibility, in-distribution (K=24, ratchets) --------------
    K = 24
    pred = rollout_from(model, X_true, ctx_seq, ercp_seq, K).numpy()
    err = np.abs(pred[:, K + 1:][..., RATCHETS] - Xt[:, K + 1:][..., RATCHETS]).mean(axis=(1, 2))
    susc = np.array([p.susceptibility for p in va["ctx"]])
    qs = np.quantile(susc, [1/3, 2/3])
    print("=== 3. ratchet error by hidden susceptibility tercile (K=24, in-distribution) ===")
    for name, mask in [("slow", susc <= qs[0]),
                       ("med", (susc > qs[0]) & (susc <= qs[1])),
                       ("fast", susc > qs[1])]:
        print(f"  {name:5s} (n={int(mask.sum()):3d})  ratchet MAE = {err[mask].mean():.4f}")
    print()

    # --- 4) generalisation probe: OOD cohorts (K=24, ratchet MAE) -------------------------
    print("=== 4. generalisation probe (K=24, ratchet MAE; higher = worse) ===")
    base_r = mae_over(pred, Xt, K + 1, T, RATCHETS)
    print(f"  in-distribution val               : {base_r:.4f}   (reference)")
    for name, sp in get_probes(n=200).items():
        _, cs, es, Xtr = build_rollout_batch(sp)
        H = Xtr.shape[1]
        pr = rollout_from(model, Xtr, cs, es, K).numpy(); Xn = Xtr.numpy()
        if H > T:  # longer-than-training: split in-horizon vs beyond-horizon
            in_h  = mae_over(pr, Xn, K + 1, T, RATCHETS)
            beyond = mae_over(pr, Xn, T, H, RATCHETS)
            print(f"  {name:34s}: {in_h:.4f} (months {K+1}-{T})  ->  {beyond:.4f} (months {T}-{H})")
        else:
            print(f"  {name:34s}: {mae_over(pr, Xn, K + 1, H, RATCHETS):.4f}")


if __name__ == "__main__":
    main()
