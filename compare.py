"""
Head-to-head: baseline (x-as-latent, memoryless)  vs  GRU-JEPA (latent-space, the naive
first attempt / dead-end, checkpoints/jepa.pt)  vs  baseline+w (native-space + GRU history latent).

NOTE: the "GRU-JEPA" column here is the minimal GRU predictor, NOT the masked TS-JEPA that the
memo reports at ~0.039. TS-JEPA trains and evaluates in-process (no checkpoint) -- run ts_jepa.py
for those numbers. Keeping them separate avoids conflating the dead-end GRU with the team's direction.

Same eval as eval.py, run for all three models on identical held-out patients and probes.
"""

import numpy as np
import torch

from data import get_split, get_probes, build_rollout_batch
from eval import rollout_from, mae_over, constraint_report, RATCHETS
from models.baseline import MonotoneStep
from models.jepa import JEPA
from models.history import HistoryStep


def load_all():
    _bk = torch.load("checkpoints/baseline.pt")
    b = MonotoneStep(hidden=_bk["hidden"], couple_m=_bk.get("couple_m", False))
    b.load_state_dict(_bk["state_dict"]); b.eval()
    j = JEPA(); j.load_state_dict(torch.load("checkpoints/jepa.pt")["state_dict"]); j.eval()
    h = HistoryStep(); h.load_state_dict(torch.load("checkpoints/history.pt")["state_dict"]); h.eval()
    return {"baseline": b, "GRU-JEPA": j, "baseline+w": h}


def roll(model, name, X, ctx, erc, K):
    if name == "baseline":
        return rollout_from(model, X, ctx, erc, K).numpy()
    return model.rollout(X, ctx, erc, K).numpy()   # JEPA / history share the signature


def main():
    models = load_all()
    _, va = get_split()
    _, ctx, erc, X = build_rollout_batch(va); Xn = X.numpy()
    names = list(models)

    # --- constraints: every model must be exactly 0 --------------------------------------
    print("=== constraint-violation rate (full free rollout) ===")
    for nm, m in models.items():
        pred = roll(m, nm, X, ctx, erc, 0)
        v, t, oob = constraint_report(pred, erc.numpy())
        print(f"  {nm:12s}  violations {v}/{t}  oob {oob}  -> {v/t:.6f}")

    # --- K-sweep, ratchet MAE ------------------------------------------------------------
    print("\n=== ratchet MAE vs conditioning window K (lower = better) ===")
    print(f"  {'K':>3} | " + " ".join(f"{n:>11}" for n in names))
    for K in (12, 24, 36):
        row = [mae_over(roll(m, nm, X, ctx, erc, K), Xn, K + 1, Xn.shape[1], RATCHETS)
               for nm, m in models.items()]
        print(f"  {K:>3} | " + " ".join(f"{r:11.4f}" for r in row))

    # --- by hidden susceptibility (K=24, in-distribution) --------------------------------
    K = 24
    susc = np.array([p.susceptibility for p in va["ctx"]])
    qs = np.quantile(susc, [1/3, 2/3])
    strata = [("slow", susc <= qs[0]), ("med", (susc > qs[0]) & (susc <= qs[1])), ("fast", susc > qs[1])]
    preds = {nm: roll(m, nm, X, ctx, erc, K) for nm, m in models.items()}
    print("\n=== ratchet MAE by hidden susceptibility (K=24) ===")
    print(f"  {'stratum':7} | " + " ".join(f"{n:>11}" for n in names))
    for sname, mask in strata:
        row = [np.abs(preds[nm][mask][:, K + 1:][..., RATCHETS] - Xn[mask][:, K + 1:][..., RATCHETS]).mean()
               for nm in names]
        print(f"  {sname:7} | " + " ".join(f"{r:11.4f}" for r in row))

    # --- generalisation probes (K=24) ----------------------------------------------------
    print("\n=== generalisation probe, ratchet MAE (K=24) ===")
    print(f"  {'cohort':26} | " + " ".join(f"{n:>11}" for n in names))
    for cname, sp in get_probes(n=200).items():
        _, cs, es, Xt = build_rollout_batch(sp); Xtn = Xt.numpy(); H = Xtn.shape[1]
        a, b = (K + 1, 60) if H > 60 else (K + 1, H)      # for long-horizon show in-horizon part
        row = [mae_over(roll(m, nm, Xt, cs, es, K), Xtn, a, b, RATCHETS) for nm, m in models.items()]
        print(f"  {cname:26} | " + " ".join(f"{r:11.4f}" for r in row))
        if H > 60:
            row2 = [mae_over(roll(m, nm, Xt, cs, es, K), Xtn, 60, H, RATCHETS) for nm, m in models.items()]
            print(f"  {'  (beyond horizon 60-96)':26} | " + " ".join(f"{r:11.4f}" for r in row2))


if __name__ == "__main__":
    main()
