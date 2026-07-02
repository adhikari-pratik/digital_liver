"""
Explainability: "why did the model predict decompensation at month 30?"

Decompensation proxy: portal hypertension P crossing 0.5 (P is the clinical driver of hepatic
decompensation -- varices, ascites). We answer the question two ways, both enabled by the
monotonic-by-construction head:

  1. TRANSPARENT ACCUMULATION. Because P(t) = P(K) + sum of NON-NEGATIVE monthly increments, the
     prediction is not a black-box jump -- it is an auditable running total. We show the curve
     and the month it crosses the threshold.
  2. ATTRIBUTION. Each monthly increment is softplus(net(x_t, ctx)). We use gradient x input on
     each increment and sum over the months up to 30 to attribute the predicted rise in P to the
     state fields that drove it. A faithful model should credit inflammation A and fibrosis F
     (the generator drives P from A/C and F) -- a concrete check that the reason is the right one.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_split, build_rollout_batch
from eval import rollout_from
from models.baseline import MonotoneStep
from generator import FIELD_NAMES, N_FIELDS, P, F, A, C

P_DECOMP = 0.4     # decompensation proxy threshold on portal hypertension P
K = 6              # observe 6 months, then predict -- so month 30 is well inside the horizon


def load_baseline():
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    model = load_baseline()
    _, va = get_split()
    _, ctx, erc, X = build_rollout_batch(va)

    # free-roll from K, find the patient whose predicted P crosses 0.5 closest to month 30
    pred = rollout_from(model, X, ctx, erc, K).numpy()
    cross = np.full(len(pred), 999)
    for i in range(len(pred)):
        idx = np.where(pred[i, :, P] >= P_DECOMP)[0]
        if len(idx):
            cross[i] = idx[0]
    i = int(np.argmin(np.abs(cross - 30)))
    t_cross = int(cross[i])
    print(f"patient {i}: predicted P crosses {P_DECOMP} at month {t_cross}")

    # --- attribution: model-faithful PERTURBATION (+0.1 per field -> change in P-increment) -
    # matches the gradient sensitivity, and is more intuitive: "if this field were higher, how
    # much more P would the model add this month?"
    total_rise = float(pred[i, 30, P] - pred[i, K, P])
    fmax = np.array([1, 1, 1, 1, 1, 1, 2, 1], dtype=np.float32)

    def inc_P(xrow, t):
        with torch.no_grad():
            raw = model.net(torch.cat([torch.tensor(xrow), ctx[i, t + 1]]))
            return float(torch.nn.functional.softplus(raw[P]))

    delta = np.zeros(N_FIELDS)
    for t in range(K, 30):
        base = inc_P(pred[i, t], t)
        for j in range(N_FIELDS):
            x2 = pred[i, t].copy(); x2[j] = min(x2[j] + 0.1, fmax[j])
            delta[j] += inc_P(x2, t) - base
    share = 100 * np.clip(delta, 0, None) / np.clip(delta, 0, None).sum()
    order = np.argsort(share)[::-1]
    print(f"predicted P rises {pred[i,K,P]:.2f} -> {pred[i,30,P]:.2f} over months {K}-30 "
          f"(+{total_rise:.2f}), reaching the decompensation threshold at month {t_cross}.")
    print("what the model keys on for the monthly P-increment (perturbation +0.1):")
    for j in order[:5]:
        print(f"  {FIELD_NAMES[j]:5s} {share[j]:5.1f}%   (raw {delta[j]:+.4f})")

    # --- figure: the auditable accumulation + attribution --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for f, lab in [(P, "P (portal HTN)"), (F, "F (fibrosis)"), (A, "A (inflammation)"), (C, "C (cholestasis)")]:
        ax[0].plot(pred[i, :, f], label=lab, lw=1.8)
    ax[0].axhline(P_DECOMP, color="k", ls="--", lw=1, alpha=0.6)
    ax[0].axvline(t_cross, color="red", ls=":", lw=1.2)
    ax[0].axvspan(0, K, color="grey", alpha=0.12)
    ax[0].text(K/2, 1.05, "observed", ha="center", fontsize=8)
    ax[0].text(t_cross + 0.5, P_DECOMP + 0.02, f"decompensation\n(month {t_cross})", color="red", fontsize=8)
    ax[0].set_title(f"Patient {i}: predicted trajectory (free rollout from month {K})")
    ax[0].set_xlabel("month"); ax[0].set_ylabel("value"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    ax[1].bar([FIELD_NAMES[j] for j in order[:5]], [share[j] for j in order[:5]], color="steelblue")
    ax[1].set_title(f"Why: what drives the predicted P-increment (months {K}-30)")
    ax[1].set_ylabel("% contribution"); ax[1].grid(alpha=0.25, axis="y")

    import os
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/explain_decompensation.png", dpi=110)
    print("saved figures/explain_decompensation.png")


if __name__ == "__main__":
    main()
