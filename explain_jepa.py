"""
Explainability, on the JEPA model itself (companion to explain.py, which runs on the baseline).

Purpose: make "the JEPA is auditable" concrete rather than asserted. The JEPA reaches the SAME
constraint head by a longer path (x -> encode -> predict-latent -> decode -> increment), but the
P-increment is still softplus(dec(zhat)[P]) added to the previous P. So the two audits the head
enables transfer verbatim:

  1. TRANSPARENT ACCUMULATION. P(t) = P(K) + sum of non-negative monthly increments -- an auditable
     running total even though the dynamics ran through a latent.
  2. ATTRIBUTION. Perturb each state field by +0.1 and read the change in the model's P-increment,
     summed over months K..30 -- "if this field were higher, how much more P would JEPA add?"

Note: this is the full JEPA (latent-prediction ON), which memo section 3 shows is the less accurate
model on this clean state. The point here is auditability, which holds regardless of accuracy.

Run: python explain_jepa.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_split, build_rollout_batch
from models.jepa import JEPA
from generator import FIELD_NAMES, N_FIELDS, FIELD_MAX, P, F, A, C

P_DECOMP = 0.4
K = 6


def main():
    m = JEPA(); m.load_state_dict(torch.load("checkpoints/jepa.pt")["state_dict"]); m.eval()
    _, va = get_split()
    _, ctx, erc, X = build_rollout_batch(va)

    with torch.no_grad():
        pred = m.rollout(X, ctx, erc, K).numpy()

    # pick the patient whose predicted P crosses the threshold closest to month 30
    cross = np.full(len(pred), 999)
    for i in range(len(pred)):
        idx = np.where(pred[i, :, P] >= P_DECOMP)[0]
        if len(idx):
            cross[i] = idx[0]
    i = int(np.argmin(np.abs(cross - 30)))
    t_cross = int(cross[i])
    print(f"patient {i}: JEPA-predicted P crosses {P_DECOMP} at month {t_cross}")

    # fixed patient latent w, inferred from the conditioning window (same as rollout semantics)
    with torch.no_grad():
        w_i = m.patient_latent(X[i:i + 1, :K + 1], ctx[i:i + 1, :K + 1])[0, K]   # [d_patient]

    fmax = np.array(FIELD_MAX, dtype=np.float32)

    def inc_P(xrow, t):
        """The model's non-negative P-increment at month t for state xrow (softplus of decode)."""
        with torch.no_grad():
            z = m.enc(torch.tensor(xrow))
            zhat = m.pred(torch.cat([z, w_i, ctx[i, t + 1]]))
            raw = m.dec(zhat)
            return float(torch.nn.functional.softplus(raw[P]))

    delta = np.zeros(N_FIELDS)
    for t in range(K, 30):
        base = inc_P(pred[i, t], t)
        for j in range(N_FIELDS):
            x2 = pred[i, t].copy(); x2[j] = min(x2[j] + 0.1, fmax[j])
            delta[j] += inc_P(x2, t) - base
    pos = np.clip(delta, 0, None)
    share = 100 * pos / (pos.sum() + 1e-9)
    order = np.argsort(share)[::-1]
    total_rise = float(pred[i, 30, P] - pred[i, K, P])
    print(f"JEPA-predicted P rises {pred[i,K,P]:.2f} -> {pred[i,30,P]:.2f} over months {K}-30 "
          f"(+{total_rise:.2f}), reaching threshold at month {t_cross}.")
    print("what the JEPA keys on for the monthly P-increment (perturbation +0.1):")
    for j in order[:5]:
        print(f"  {FIELD_NAMES[j]:5s} {share[j]:5.1f}%   (raw {delta[j]:+.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for f, lab in [(P, "P (portal HTN)"), (F, "F (fibrosis)"), (A, "A (inflammation)"), (C, "C (cholestasis)")]:
        ax[0].plot(pred[i, :, f], label=lab, lw=1.8)
    ax[0].axhline(P_DECOMP, color="k", ls="--", lw=1, alpha=0.6)
    ax[0].axvline(t_cross, color="red", ls=":", lw=1.2)
    ax[0].axvspan(0, K, color="grey", alpha=0.12)
    ax[0].text(K / 2, 1.05, "observed", ha="center", fontsize=8)
    ax[0].text(t_cross + 0.5, P_DECOMP + 0.02, f"decompensation\n(month {t_cross})", color="red", fontsize=8)
    ax[0].set_title(f"JEPA, patient {i}: predicted trajectory (free rollout from month {K})")
    ax[0].set_xlabel("month"); ax[0].set_ylabel("value"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    ax[1].bar([FIELD_NAMES[j] for j in order[:5]], [share[j] for j in order[:5]], color="crimson")
    ax[1].set_title(f"Why (JEPA): what drives the predicted P-increment (months {K}-30)")
    ax[1].set_ylabel("% contribution"); ax[1].grid(alpha=0.25, axis="y")

    import os
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/explain_decompensation_jepa.png", dpi=110)
    print("saved figures/explain_decompensation_jepa.png")


if __name__ == "__main__":
    main()
