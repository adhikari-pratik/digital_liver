"""
Does training TS-JEPA for real-world conditions (sensor noise, sparse/stale visits) let it BEAT the
baseline on those axes -- honestly, with an eval behind every number?

The CLEAN-trained JEPA loses the noise axis (noise_robustness.py) and hits an OOD cliff at very stale
visits (missing_visits.py, K0<8) -- both because it was trained on CLEAN, contiguous data. The
principled fix is the JEPA recipe itself: train with corrupted / dropped views so the encoder LEARNS
to denoise and to bridge gaps (the EMA target still sees the clean full sequence, so the objective
pulls noisy/sparse -> clean). We train three variants and score them vs the baseline.

  JEPA-clean : shipped recipe (contiguous future-mask, clean)              -> the memo's 0.039 model
  JEPA-drop  : + random visit-dropout in history + shorter windows (kmin)  -> robust to sparse/stale
  JEPA-noise : + Gaussian sensor noise on the online view                  -> learns to denoise

NOTHING is fabricated: each variant is trained and measured on the SAME held-out patients. Whatever
the grid shows is what we report -- including a baseline win.
"""
import numpy as np
import torch

from data import get_split, build_rollout_batch, T
from eval import rollout_from, mae_over, RATCHETS
from models.baseline import MonotoneStep
from generator import FIELD_MAX
import ts_jepa as tj

FMAX = np.array(FIELD_MAX, dtype=np.float32)
SCORE_A, SCORE_B = 25, T
K0S = [24, 18, 12, 9, 6]           # staleness sweep (incl. the K0<8 OOD region for clean-JEPA)
SIGMAS = [0.0, 0.05, 0.10, 0.15]   # sensor-noise sweep at a fresh visit (K0=24)
SEED = 0


def load_baseline():
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def jepa_forecast(mt, Xv, cv, ev, K0):
    with torch.no_grad():
        return tj.decode_forecast(
            mt.dec, mt.enc(Xv, cv, tj.obs_mask(Xv.shape[0], K0)), Xv, ev, K0).numpy()


def add_noise(Xn, sigma, K0, rng):
    """Corrupt the observed window [0..K0] with clamped Gaussian noise; keep the future clean."""
    Xc = Xn.copy()
    if sigma > 0:
        w = Xc[:, :K0 + 1]
        Xc[:, :K0 + 1] = np.clip(w + rng.normal(0, sigma, w.shape).astype(np.float32), 0.0, FMAX)
    return torch.tensor(Xc)


def best_of(*named):
    return min(named, key=lambda t: t[1])[0]


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    base = load_baseline()

    print("  training JEPA-clean ...", flush=True)
    *_, m_clean = tj.train(seed=SEED, return_model=True)
    print("  training JEPA-drop  (drop_p=0.3, kmin=3) ...", flush=True)
    *_, m_drop = tj.train(seed=SEED, drop_p=0.3, kmin=3, return_model=True)
    print("  training JEPA-noise (noise_std=0.08) ...", flush=True)
    *_, m_noise = tj.train(seed=SEED, noise_std=0.08, return_model=True)

    # ---- sweep A: stale last visit (clean observations) -------------------------------------
    print(f"\n=== A) STALE LAST VISIT: ratchet MAE on [{SCORE_A}..{SCORE_B}] (clean obs) ===")
    print(f"  {'K0 (gap)':>13} | {'baseline':>9} | {'JEPA-clean':>10} | {'JEPA-drop':>9} | best")
    for K0 in K0S:
        b = mae_over(rollout_from(base, Xv, cv, ev, K0).numpy(), Xn, SCORE_A, SCORE_B, RATCHETS)
        jc = mae_over(jepa_forecast(m_clean, Xv, cv, ev, K0), Xn, SCORE_A, SCORE_B, RATCHETS)
        jd = mae_over(jepa_forecast(m_drop, Xv, cv, ev, K0), Xn, SCORE_A, SCORE_B, RATCHETS)
        g = 24 - K0
        best = best_of(("baseline", b), ("JEPA-clean", jc), ("JEPA-drop", jd))
        print(f"  K0={K0:>2} (G={g:>2}mo) | {b:>9.4f} | {jc:>10.4f} | {jd:>9.4f} | {best}")

    # ---- sweep B: sensor noise at a fresh visit ---------------------------------------------
    rng = np.random.default_rng(SEED)
    print(f"\n=== B) SENSOR NOISE at fresh visit K0=24: ratchet MAE on [{SCORE_A}..{SCORE_B}] (vs CLEAN truth) ===")
    print(f"  {'sigma':>6} | {'baseline':>9} | {'JEPA-clean':>10} | {'JEPA-noise':>10} | best")
    for s in SIGMAS:
        Xc = add_noise(Xn, s, 24, rng)
        b = mae_over(rollout_from(base, Xc, cv, ev, 24).numpy(), Xn, SCORE_A, SCORE_B, RATCHETS)
        jc = mae_over(jepa_forecast(m_clean, Xc, cv, ev, 24), Xn, SCORE_A, SCORE_B, RATCHETS)
        jn = mae_over(jepa_forecast(m_noise, Xc, cv, ev, 24), Xn, SCORE_A, SCORE_B, RATCHETS)
        best = best_of(("baseline", b), ("JEPA-clean", jc), ("JEPA-noise", jn))
        print(f"  {s:>6.2f} | {b:>9.4f} | {jc:>10.4f} | {jn:>10.4f} | {best}")

    print("\n  Honest read: report the grid as-is. JEPA should 'win' only where its trained-for")
    print("  condition (stale/sparse history, sensor noise) actually appears; the baseline owns the")
    print("  clean, fresh, fully-observed case. Anchor noise is shared by both, so the noise-axis gain")
    print("  is capped by construction (cumsum-from-observed) -- flagged, not hidden.")


if __name__ == "__main__":
    main()
