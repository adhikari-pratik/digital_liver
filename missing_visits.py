"""
Does TS-JEPA forecast better than the memoryless baseline when the last visit is STALE?
(the 'Missing Visit Rate' axis -- the one place JEPA's hidden state should structurally help)

Setup: the patient was last seen at month K0; every month after K0 is missing. Both models forecast
the SAME fixed window [25..T] against the CLEAN truth, and we sweep the staleness gap G = 24 - K0
(0, 3, ... months of missing recent visits). The scored window is held fixed so only the anchor's
staleness varies.

Why this is the FAIR test (and random interior dropout is not): TS-JEPA was trained on CONTIGUOUS
future-masking (obs_mask = [0..K] True, K random). A stale last visit is exactly that pattern -->
in-distribution for the encoder. Random scattered gaps would be OUT of its training distribution and
would handicap it; a fair random-dropout test needs dropout-augmented training (a next step, noted).

Mechanism under test: the baseline is memoryless -- it can only anchor on x[K0] and compound forward
blind. TS-JEPA attends over the whole pre-gap history [0..K0]. If integrating that history buys
anything, JEPA should degrade MORE GRACEFULLY as the last visit gets staler.
"""
import numpy as np
import torch

from data import get_split, build_rollout_batch, T
from eval import rollout_from, mae_over, RATCHETS
from models.baseline import MonotoneStep

SCORE_A, SCORE_B = 25, T          # fixed scored window for every gap
K0S = [24, 21, 18, 15, 12, 9, 6]  # last-visit month; gap G = 24 - K0
SEEDS = [0, 1, 2]                 # confirm the crossover is seed-stable, not a fluke
TRAIN_KMIN = 8                    # ts_jepa trains masks with K in [8,40]; K0<8 is OOD for the encoder


def load_baseline():
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def tsjepa_sweep(seed, Xv, cv, ev, Xn):
    """Train one TS-JEPA; return its ratchet MAE at each staleness K0."""
    import ts_jepa as tj
    print(f"  training TS-JEPA seed {seed}...", flush=True)
    *_, mt = tj.train(seed=seed, return_model=True)
    out = {}
    for K0 in K0S:
        with torch.no_grad():
            rj = tj.decode_forecast(
                mt.dec, mt.enc(Xv, cv, tj.obs_mask(Xv.shape[0], K0)), Xv, ev, K0).numpy()
        out[K0] = mae_over(rj, Xn, SCORE_A, SCORE_B, RATCHETS)
    return out


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    base = load_baseline()

    # baseline is deterministic: one MAE per staleness
    bmae = {}
    for K0 in K0S:
        with torch.no_grad():
            rb = rollout_from(base, Xv, cv, ev, K0).numpy()
        bmae[K0] = mae_over(rb, Xn, SCORE_A, SCORE_B, RATCHETS)

    # TS-JEPA: mean +/- sd over seeds
    js = [tsjepa_sweep(s, Xv, cv, ev, Xn) for s in SEEDS]
    jmean = {K0: float(np.mean([j[K0] for j in js])) for K0 in K0S}
    jsd = {K0: float(np.std([j[K0] for j in js])) for K0 in K0S}

    print(f"\n=== ratchet MAE on fixed window [{SCORE_A}..{SCORE_B}] vs staleness of last visit "
          f"(TS-JEPA over {len(SEEDS)} seeds) ===")
    print(f"  last visit K0 (gap)  | {'baseline':>9} | {'TS-JEPA (mean+/-sd)':>20} | degrades less")
    b0, j0 = bmae[24], jmean[24]
    for K0 in K0S:
        db, dj = bmae[K0] - b0, jmean[K0] - j0
        less = "TS-JEPA" if dj < db else "baseline"
        wins = "  <-- JEPA wins outright" if jmean[K0] < bmae[K0] else ""
        ood = "  (OOD: K0<8)" if K0 < TRAIN_KMIN else ""
        g = 24 - K0
        print(f"  K0={K0:>2} (G={g:>2}mo){ood:>12} | {bmae[K0]:>9.4f} | "
              f"{jmean[K0]:>7.4f} +/- {jsd[K0]:.4f}    | {less}   "
              f"(deg base +{db:.4f}, jepa +{dj:.4f}){wins}")
    print("\n  'degrades less' = smaller MAE increase vs its own K0=24 (fresh-visit) baseline.")
    print("  In-distribution (K0>=8), if TS-JEPA degrades less and crosses over, its hidden-state")
    print("  integration earns its keep -- the one structural JEPA advantage this clean toy exposes.")
    print("  K0=6 is OOD (encoder trained on masks K in [8,40]); read it as an architectural cap, not")
    print("  a refutation.")


if __name__ == "__main__":
    main()
