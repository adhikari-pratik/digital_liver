"""
Does TS-JEPA actually beat the baseline under CLINICAL/GEOMETRIC metrics (not just point MAE)?

A reviewer argued point-step MAE unfairly favors a conservative baseline and that shape/ranking/
distribution metrics would reveal TS-JEPA's superiority. That is a TESTABLE claim -- so test it,
don't assume it. Championing TS-JEPA on these metrics is only honest if it actually wins them.

Head-to-head on the SAME held-out patients, free rollout from K=24:
  1. Point MAE (ratchets)         -- the existing metric (lower better)
  2. DTW distance (ratchets)      -- shape/sequence match, tolerant of time-shift (lower better)
  3. C-index (cirrhosis ranking)  -- does predicted risk rank true time-to-cirrhosis? (higher better)
  4. Cohort Wasserstein-1 (F_final) -- does the cohort outcome DISTRIBUTION match truth? (lower better)
"""
import numpy as np
import torch

from data import get_split, build_rollout_batch, T
from eval import rollout_from, mae_over, RATCHETS
from models.baseline import MonotoneStep
from generator import F as F_IDX

K = 24
CIRRH = 0.8           # cirrhosis onset threshold (aligned with derived.py / the other eval scripts)


def dtw(a, b):
    """Normalised DTW distance between two [L,D] sequences (Euclidean local cost)."""
    L, M = len(a), len(b)
    D = np.full((L + 1, M + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, L + 1):
        ai = a[i - 1]
        for j in range(1, M + 1):
            cost = np.linalg.norm(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[L, M] / (L + M)


def mean_dtw(pred, true, cols):
    return float(np.mean([dtw(pred[n][:, cols], true[n][:, cols]) for n in range(pred.shape[0])]))


def c_index(risk, event_time, observed):
    """Harrell's C: do higher risk scores rank earlier events first?"""
    n = len(risk); num = den = 0.0
    for i in range(n):
        if not observed[i]:
            continue
        for j in range(n):
            if event_time[j] > event_time[i]:            # i has the earlier event
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / den if den else float("nan")


def event_times(X):
    """First month F crosses CIRRH (else censored at T)."""
    F = X[:, :, F_IDX]
    reached = F >= CIRRH
    obs = reached.any(1)
    t = np.where(reached.any(1), reached.argmax(1), T)
    return t, obs


def w1(a, b):
    """1-D Wasserstein-1 between two equal-size sample sets."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def baseline_rollout(Xv, cv, ev):
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return rollout_from(m, Xv, cv, ev, K).numpy()


def tsjepa_rollout(Xv, cv, ev):
    import ts_jepa as tj
    print("  training a TS-JEPA (seed 0) for the comparison...", flush=True)
    *_, mt = tj.train(seed=0, return_model=True)
    with torch.no_grad():
        return tj.decode_forecast(
            mt.dec, mt.enc(Xv, cv, tj.obs_mask(Xv.shape[0], tj.K_EVAL)), Xv, ev, tj.K_EVAL).numpy()


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    et, obs = event_times(Xn); trueF = Xn[:, -1, F_IDX]

    rolls = {"baseline": baseline_rollout(Xv, cv, ev), "TS-JEPA": tsjepa_rollout(Xv, cv, ev)}
    win = "forecast window only, K+1..T"
    print(f"\n=== baseline vs TS-JEPA under four metrics ({win}) ===")
    print(f"  {'metric':28} | {'baseline':>10} | {'TS-JEPA':>10} | better")
    rows = {}
    for name, pred in rolls.items():
        mae = mae_over(pred, Xn, K + 1, T, RATCHETS)
        dtw_r = mean_dtw(pred[:, K + 1:], Xn[:, K + 1:], RATCHETS)
        ci = c_index(pred[:, -1, F_IDX], et, obs)                     # risk = predicted final F
        ww = w1(pred[:, -1, F_IDX], trueF)
        rows[name] = (mae, dtw_r, ci, ww)
    def line(lbl, key, hi=False):
        b, j = rows["baseline"][key], rows["TS-JEPA"][key]
        better = "TS-JEPA" if (j > b if hi else j < b) else "baseline"
        print(f"  {lbl:28} | {b:>10.4f} | {j:>10.4f} | {better}")
    line("1. point MAE (ratchets)", 0)
    line("2. DTW distance (ratchets)", 1)
    line("3. C-index (cirrhosis rank)", 2, hi=True)
    line("4. Wasserstein-1 (F_final)", 3)
    print(f"\n  true cirrhosis-onset base rate: {int(obs.sum())}/{len(obs)}; true mean final F = {trueF.mean():.3f}")


if __name__ == "__main__":
    main()
