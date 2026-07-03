"""
Precompute REAL rollouts for the interactive demo (no fabrication): for a handful of held-out
patients, across a grid of sensor-noise sigma x months-since-last-visit, roll out both the memoryless
baseline and the denoised-anchor TS-JEPA and record the fibrosis-F trajectories + ratchet MAE. The
interactive page embeds this JSON and just looks values up as the sliders move.

Output: demo_data.json in the repo root (seed 0; same models/eval as jepa_denoise.py / figures_showcase.py)
"""
import json
import os
import numpy as np
import torch

from data import get_split, build_rollout_batch
from eval import rollout_from, mae_over, RATCHETS
from generator import F as F_IDX
import ts_jepa as tj
import jepa_denoise as jd

T = 60
SIGMAS = [0.0, 0.05, 0.10, 0.15]
K0S = [24, 18, 12, 9]              # months since last visit = 24 - K0 -> [0, 6, 12, 15]
N_PATIENTS = 5
R3 = lambda a: [round(float(x), 3) for x in a]


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    base = jd.load_baseline()
    print("  training denoised-anchor TS-JEPA (seed 0)...", flush=True)
    m = jd.train(seed=0)

    # pick progressors spanning slow->severe (visible dynamics), by true final F
    finalF = Xn[:, -1, F_IDX]
    cand = np.where(finalF > 0.35)[0]
    cand = cand[np.argsort(finalF[cand])]
    pick = cand[np.linspace(0, len(cand) - 1, N_PATIENTS).round().astype(int)]
    labels = ["slow", "moderate", "advancing", "fast", "severe"]

    patients = [{"id": int(p), "label": labels[i], "finalF": round(float(finalF[p]), 3),
                 "truthF": R3(Xn[p, :, F_IDX])} for i, p in enumerate(pick)]

    grid = []  # grid[sigma_idx][gap_idx][patient_idx]
    for s in SIGMAS:
        rng = np.random.default_rng(int(s * 1000))     # deterministic per sigma
        row = []
        for K0 in K0S:
            Xc = jd.add_noise(Xn, s, K0, rng)
            with torch.no_grad():
                rb = rollout_from(base, Xc, cv, ev, K0).numpy()
                z = m.enc(Xc, cv, tj.obs_mask(Xc.shape[0], K0))
                out, state_est = jd.decode_denoised(m, z, Xc, ev, K0, denoised=True)
            rj = out.numpy(); anc = state_est.numpy()
            Xcn = Xc.numpy()
            cell = []
            for p in pick:
                bmae = mae_over(rb[p:p + 1], Xn[p:p + 1], K0 + 1, T, RATCHETS)
                jmae = mae_over(rj[p:p + 1], Xn[p:p + 1], K0 + 1, T, RATCHETS)
                cell.append({
                    "K0": int(K0),
                    "obsF": R3(Xcn[p, :K0 + 1, F_IDX]),        # noisy observations 0..K0
                    "baseF": R3(rb[p, :, F_IDX]),              # baseline rollout (full)
                    "jepaF": R3(rj[p, :, F_IDX]),              # JEPA rollout (full)
                    "anchor": round(float(anc[p, K0, F_IDX]), 3),  # JEPA denoised current-state anchor
                    "baseMAE": round(float(bmae), 4),
                    "jepaMAE": round(float(jmae), 4),
                })
            row.append(cell)
        grid.append(row)

    data = {"sigmas": SIGMAS, "gaps": [24 - k for k in K0S], "K0s": K0S,
            "T": T, "patients": patients, "grid": grid}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  wrote {out_path}  ({len(json.dumps(data))//1024} KB)")
    # quick sanity: at max stress, JEPA should beat baseline for most patients
    c = grid[-1][-1]
    wins = sum(1 for e in c if e["jepaMAE"] < e["baseMAE"])
    print(f"  sanity @ sigma=0.15,gap=15mo: JEPA beats baseline on {wins}/{N_PATIENTS} patients")


if __name__ == "__main__":
    main()
