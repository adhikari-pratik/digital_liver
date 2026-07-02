"""
FAST verification of the MDN distributional head (§8, D23) from a saved checkpoint -- no training.

Loads checkpoints/mdn.pt (written by `python mdn_forecast.py`, the first seed) and reports the tail
recall + calibration in a few seconds, so a reviewer can check the claim without the ~5-min 3-seed run.

HONEST NOTE: this is ONE seed. The headline numbers in memo §1/§8 are the 3-SEED AGGREGATE
(ratchet MAE 0.028 +/- 0.002, cirrhosis recall @q90 0.82 +/- 0.10, coverage 0.70 +/- 0.15) -- run
`python mdn_forecast.py` to reproduce that. This script shows the single saved seed for a quick look;
per-seed calibration is deliberately variable (that is the D25 finding), so expect this seed to differ
from the mean.
"""
import numpy as np
import torch

from mdn_forecast import MDNStep, mc_rollout, mean_rollout, K_OBS, S_SAMP, CIRRH
from data import get_split, build_rollout_batch, T
from eval import mae_over, RATCHETS


def main():
    torch.manual_seed(0)                 # deterministic MC sampling -> reproducible numbers
    ck = torch.load("checkpoints/mdn.pt")
    m = MDNStep(hidden=ck["hidden"], n_mix=ck["n_mix"], couple_m=ck["couple_m"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    print(f"loaded checkpoints/mdn.pt (seed {ck.get('seed', '?')}, n_mix={ck['n_mix']}) -- single seed\n")

    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    acc = mae_over(mean_rollout(m, Xv, cv, ev, K_OBS), Xn, K_OBS + 1, T, RATCHETS)
    Fsamp = mc_rollout(m, Xv, cv, ev, K_OBS, S_SAMP).numpy()
    trueF = Xn[:, -1, 0]; true_cirr = trueF >= CIRRH
    q = lambda p: np.quantile(Fsamp, p, axis=0)
    lo, hi, q90 = q(0.05), q(0.95), q(0.90)

    def rp(flag):
        tp = int((flag & true_cirr).sum()); fp = int((flag & ~true_cirr).sum()); fn = int((~flag & true_cirr).sum())
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    p90, r90 = rp(q90 >= CIRRH); p95, r95 = rp(hi >= CIRRH)
    cov = float(((trueF >= lo) & (trueF <= hi)).mean())

    print(f"  ratchet MAE (K={K_OBS}, mixture-mean) = {acc:.4f}   (point baseline 0.033)")
    print(f"  cirrhosis recall @q90 = {r90:.2f}  prec {p90:.2f}   |   @q95 = {r95:.2f}  prec {p95:.2f}"
          f"   (point baseline recall 0.27)")
    print(f"  90% interval coverage = {cov:.2f}")
    print("\n  3-seed aggregate (memo sec 1/8, run mdn_forecast.py): MAE 0.028+/-0.002, "
          "recall@q90 0.82+/-0.10, coverage 0.70+/-0.15.")


if __name__ == "__main__":
    main()
