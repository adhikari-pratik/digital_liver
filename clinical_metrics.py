"""
Decision-relevant metrics beyond value-MAE -- the numbers a clinical team actually asks for.

  1. EVENT-TIMING error: for decompensation (P>=0.4) and cirrhosis onset (F>=0.8), how far off is
     the PREDICTED month of crossing vs the TRUE month? Reports recall (did we catch the event at
     all), miss rate, timing bias (+ = we predict it LATE), and false-alarm rate. For a world model
     used in decisions, WHEN an event happens matters more than the exact field value.
  2. CIRRHOSIS classification: treat predicted final fibrosis as a risk score; report AUC +
     precision/recall at the clinical threshold, and a risk-stratification (reliability) table.
  3. POPULATION-DISTRIBUTION fidelity: does the SIMULATED cohort match the TRUE cohort's outcome
     distribution (Wasserstein-1 on final F/M; true-vs-predicted event rates)? The "is the twin
     realistic at population scale" check -- on-theme for a world model.
  4. UNCERTAINTY (light, honest): prediction intervals from held-out residual quantiles + empirical
     coverage. NOTE: homoscedastic / first-order -- true per-input calibration needs a distributional
     head or ensemble (named as next-step, not built).

Shipped baseline. Run: python clinical_metrics.py
"""
import numpy as np
import torch

from data import get_split, build_rollout_batch
from eval import load_model, rollout_from
from derived import is_cirrhotic, CIRRHOSIS_F
from generator import F, P, M, FIELD_NAMES

K_OBS = 12          # observe 0..12, predict the rest -> events after month 12 are genuine predictions
P_DECOMP = 0.4      # decompensation proxy on portal hypertension


def onset(traj, thresh):
    idx = np.where(traj >= thresh)[0]
    return int(idx[0]) if len(idx) else -1


def auc(label, score):
    label = np.asarray(label, bool); n1 = label.sum(); n0 = (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = score.argsort(); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def w1(a, b):
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def timing(true, pred, f, thresh, name):
    n = true.shape[0]
    to = np.array([onset(true[i, :, f], thresh) for i in range(n)])
    po = np.array([onset(pred[i, :, f], thresh) for i in range(n)])
    ev = to > K_OBS                                   # true event in the PREDICTED region
    n_ev = int(ev.sum())
    caught = (po[ev] >= 0)                            # model predicts a crossing somewhere
    err = po[ev][caught] - to[ev][caught]            # signed months (+ = predicted LATE)
    never = (to == -1)                               # patients who never truly cross
    false_alarm = int((po[never] > K_OBS).sum())
    print(f"  {name}:  true events (after m{K_OBS}) = {n_ev}")
    if n_ev:
        print(f"     recall (caught)      = {caught.sum()}/{n_ev} = {caught.mean():.2f}")
        print(f"     missed (real, unflagged) = {n_ev - caught.sum()}")
        if caught.sum():
            print(f"     timing error |months|: median={np.median(np.abs(err)):.1f}  "
                  f"mean signed={err.mean():+.1f}  (+ = predicted late)")
    print(f"     false alarms (no true event, model flags one) = {false_alarm}/{int(never.sum())}\n")


def main():
    m = load_model()
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va)
    true = Xv.numpy()
    pred = rollout_from(m, Xv, cv, ev, K_OBS).numpy()

    print(f"=== 1. EVENT-TIMING error (observe 0..{K_OBS}, predict the rest) ===")
    timing(true, pred, P, P_DECOMP, "decompensation (P>=0.4)")
    timing(true, pred, F, CIRRHOSIS_F, "cirrhosis onset (F>=0.8)")

    print("=== 2. CIRRHOSIS classification (becomes cirrhotic by end of horizon) ===")
    label = is_cirrhotic(true[:, -1, F])
    score = pred[:, -1, F]
    prd = score >= CIRRHOSIS_F
    tp = int((prd & label).sum()); fp = int((prd & ~label).sum()); fn = int((~prd & label).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"  positives (true cirrhotic) = {int(label.sum())}/{len(label)}")
    print(f"  AUC (pred final F as risk score) = {auc(label, score):.3f}")
    print(f"  at threshold F>=0.8:  precision={prec:.2f}  recall={rec:.2f}  (tp={tp} fp={fp} fn={fn})")
    print("  risk stratification (predicted final F bin -> observed cirrhosis rate):")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        b = (score >= lo) & (score < lo + 0.2)
        if b.sum():
            print(f"     F_pred [{lo:.1f},{lo+0.2:.1f}):  n={int(b.sum()):3d}  observed cirrhotic={label[b].mean():.2f}")
    print()

    print("=== 3. POPULATION-DISTRIBUTION fidelity (final month, true vs simulated cohort) ===")
    for f, nm in [(F, "F"), (M, "M")]:
        print(f"  Wasserstein-1({nm}_final): {w1(true[:, -1, f], pred[:, -1, f]):.4f}  "
              f"(true mean {true[:, -1, f].mean():.3f} vs pred {pred[:, -1, f].mean():.3f})")
    for lab, f, th in [("cirrhotic", F, CIRRHOSIS_F), ("decompensated", P, P_DECOMP)]:
        tr = float((true[:, -1, f] >= th).mean()); pr = float((pred[:, -1, f] >= th).mean())
        print(f"  {lab} rate: true={tr:.3f}  predicted={pr:.3f}  (gap {pr - tr:+.3f})")
    print()

    print("=== 4. UNCERTAINTY: residual-based 90% intervals + empirical coverage (homoscedastic) ===")
    half = pred.shape[0] // 2
    cal_r = np.abs(pred[:half, K_OBS + 1:] - true[:half, K_OBS + 1:])       # calibration residuals
    hw = np.quantile(cal_r.reshape(-1, 8), 0.90, axis=0)                    # per-field 90% halfwidth
    test_r = np.abs(pred[half:, K_OBS + 1:] - true[half:, K_OBS + 1:])
    cov = (test_r <= hw).reshape(-1, 8).mean(axis=0)                        # empirical coverage
    print("  per-field 90% interval coverage on held-out (nominal 0.90):")
    for i in [F, P, M]:
        print(f"     {FIELD_NAMES[i]:5s} halfwidth={hw[i]:.3f}  coverage={cov[i]:.2f}")
    print("  NOTE: homoscedastic first-order intervals; per-input calibration needs a "
          "distributional head / ensemble (next step, not built).")


if __name__ == "__main__":
    main()
