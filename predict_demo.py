"""
Concrete demonstration of the core task: given a patient's trajectory SO FAR (months 0..K),
predict the FUTURE (months K+1..T). Saves figures/prediction_example.png.
"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import get_split, build_rollout_batch
from models.baseline import MonotoneStep
from eval import rollout_from
from generator import FIELD_NAMES, F, P, M, A

K = 24
_, va = get_split()
_, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
ck = torch.load("checkpoints/baseline.pt"); m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False)); m.load_state_dict(ck["state_dict"]); m.eval()
pred = rollout_from(m, Xv, cv, ev, K).numpy()

i = 12  # a representative held-out patient
fig, ax = plt.subplots(figsize=(9, 5))
for f, col in [(F, "tab:blue"), (P, "tab:orange"), (M, "tab:green"), (A, "tab:red")]:
    ax.plot(range(K + 1), Xn[i, :K + 1, f], color=col, lw=2)                       # observed
    ax.plot(range(K, 60), Xn[i, K:, f], color=col, lw=2, alpha=0.35)               # true future (faint)
    ax.plot(range(K, 60), pred[i, K:, f], color=col, lw=2, ls="--")               # PREDICTED future
    ax.text(60.3, pred[i, -1, f], FIELD_NAMES[f], color=col, fontsize=9, va="center")
ax.axvspan(0, K, color="grey", alpha=0.12); ax.text(K/2, 1.03, "observed (given)", ha="center", fontsize=9)
ax.text((K+60)/2, 1.03, "PREDICTED future  (dashed = model, faint = truth)", ha="center", fontsize=9)
ax.axvline(K, color="k", ls=":", lw=1)
mae = np.abs(pred[i, K+1:] - Xn[i, K+1:]).mean()
ax.set_title(f"Patient {i}: predict the future from month {K} onward   (mean abs error {mae:.3f})")
ax.set_xlabel("month"); ax.set_ylabel("clinical state value"); ax.set_ylim(-0.05, 1.12); ax.grid(alpha=0.25)
import os; os.makedirs("figures", exist_ok=True)
fig.tight_layout(); fig.savefig("figures/prediction_example.png", dpi=110)
print(f"saved figures/prediction_example.png  (patient {i}, future-prediction MAE {mae:.3f})")
