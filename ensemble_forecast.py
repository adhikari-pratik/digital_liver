"""
Probabilistic forecasting experiment (§8): does a DEEP ENSEMBLE of the multistep baseline recover
the cirrhosis tail the single point estimate misses, and give calibrated per-patient uncertainty?

Hypothesis to test honestly: ensembling captures EPISTEMIC uncertainty, but the tail miss here is
largely ALEATORIC (hidden susceptibility + random flares -> the same short history admits multiple
futures). If so, the ensemble spread will be too NARROW to catch the tail -> the fix is a
distributional/generative head, not just ensembling. Measure, don't assume.

Trains M baselines (different seeds), rolls each out (K=24), and reports:
  - accuracy: single vs ensemble-mean ratchet MAE
  - tail recall: cirrhosis (final F>=0.8) via single / ens-mean / ens-max / ens-upper(1.64sd)
  - calibration: coverage of the ensemble 90% interval on final F, and its width
"""
import numpy as np
import torch

from data import get_split, build_rollout_batch, T
from models.baseline import MonotoneStep
from eval import rollout_from, mae_over, RATCHETS

EPOCHS, PBATCH, LR, HIDDEN, L, K, M = 90, 16, 1e-3, 64, 6, 24, 5


def train_one(seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, _ = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    n = X.shape[0]
    model = MonotoneStep(hidden=HIDDEN, couple_m=True)   # match the shipped baseline (D20)
    opt = torch.optim.Adam(model.parameters(), lr=LR); mse = torch.nn.MSELoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, PBATCH):
            b = perm[i:i + PBATCH]; Xb, cb, eb = X[b], ctx[b], erc[b]
            loss = mse(model(Xb[:, :-1], cb[:, 1:], eb[:, 1:]), Xb[:, 1:])
            s = int(rng.integers(0, T - 1 - L)); cur = Xb[:, s]
            for k in range(L):
                cur = model(cur, cb[:, s + k + 1], eb[:, s + k + 1]); loss = loss + mse(cur, Xb[:, s + k + 1]) / L
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    rolls = []
    for sd in range(M):
        m = train_one(sd)
        with torch.no_grad():
            rolls.append(rollout_from(m, Xv, cv, ev, K).numpy())
        print(f"  trained ensemble member seed {sd}", flush=True)
    R = np.stack(rolls)                       # [M,N,T,8]
    ens_mean = R.mean(0)

    print("\n=== accuracy: does averaging help? ===")
    print(f"  single (seed0) ratchet MAE (K=24) = {mae_over(R[0], Xn, K + 1, T, RATCHETS):.4f}")
    print(f"  ensemble-mean  ratchet MAE (K=24) = {mae_over(ens_mean, Xn, K + 1, T, RATCHETS):.4f}")

    trueF = Xn[:, -1, 0]; true_cirr = trueF >= 0.8
    predF_single = R[0][:, -1, 0]
    predF_mean = ens_mean[:, -1, 0]
    predF_max = R.max(0)[:, -1, 0]
    predF_std = R[:, :, -1, 0].std(0)
    predF_upper = predF_mean + 1.64 * predF_std

    def rp(flag):
        tp = int((flag & true_cirr).sum()); fp = int((flag & ~true_cirr).sum()); fn = int((~flag & true_cirr).sum())
        return tp, fp, fn, tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    print(f"\n=== cirrhosis tail (true {int(true_cirr.sum())}/{len(true_cirr)}): does the ensemble catch it? ===")
    for name, pf in [("single      F>=0.8", predF_single >= 0.8),
                     ("ens-mean    F>=0.8", predF_mean >= 0.8),
                     ("ens-max     F>=0.8", predF_max >= 0.8),
                     ("ens-upper1.64sd>=.8", predF_upper >= 0.8)]:
        tp, fp, fn, prec, rec = rp(pf)
        print(f"  {name:20} recall={rec:.2f} precision={prec:.2f} (tp={tp} fp={fp} fn={fn})")

    lo = predF_mean - 1.64 * predF_std; hi = predF_mean + 1.64 * predF_std
    cov = float(((trueF >= lo) & (trueF <= hi)).mean())
    print(f"\n=== calibration (final F) ===")
    print(f"  ensemble 90% interval coverage = {cov:.2f} (nominal 0.90); mean halfwidth = {1.64 * predF_std.mean():.3f}")
    print(f"  mean per-patient ensemble std(final F) = {predF_std.mean():.3f}  (small => spread is epistemic-only)")
    # is the spread even pointed at the right patients? corr(std, |error|)
    err = np.abs(predF_mean - trueF)
    c = float(np.corrcoef(predF_std, err)[0, 1])
    print(f"  corr(ensemble std, |error|) = {c:.2f}  (>0 => uncertainty at least tracks where it errs)")


if __name__ == "__main__":
    main()
