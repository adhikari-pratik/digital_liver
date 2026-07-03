"""
Does TS-JEPA denoise better than the raw baseline? (the assignment's core "what does JEPA buy" claim)

A reviewer argued: inject hospital observation noise into a held-out patient's input stream and the
raw baseline compounds the sensor error while TS-JEPA's encoder projects onto a clean manifold. Test it.

Setup: add Gaussian observation noise (sigma * N(0,1), clamped to valid ranges) to the OBSERVED window
x[0..K] of held-out patients, free-roll both models from K, and measure ratchet MAE against the CLEAN
true future x[K+1..T]. Sweep sigma. If TS-JEPA degrades LESS, the denoising claim holds here.

HONEST CAVEAT (stated before running): both models are trained on CLEAN data, so neither *learned* to
denoise; a true denoising encoder needs noise-augmented training (VICReg invariance to corrupted views).
This tests robustness-to-shift, the reviewer's literal proposal — not learned denoising. Cf. D12, where
the fuller noise-substrate experiment was muddy (JEPA won low/mid sigma, lost at high sigma).
"""
import numpy as np
import torch

from data import get_split, build_rollout_batch, T
from eval import rollout_from, mae_over, RATCHETS
from models.baseline import MonotoneStep
from generator import FIELD_MAX

K = 24
SIGMAS = [0.0, 0.05, 0.10, 0.15]
FMAX = np.array(FIELD_MAX, dtype=np.float32)


def load_baseline():
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def train_tsjepa():
    import ts_jepa as tj
    print("  training a TS-JEPA (seed 0)...", flush=True)
    *_, mt = tj.train(seed=0, return_model=True)
    return mt, tj


def noised(Xn, sigma, rng):
    """Corrupt the observed window [0..K] with clamped Gaussian noise; keep the future clean."""
    Xc = Xn.copy()
    if sigma > 0:
        noise = rng.normal(0, sigma, Xc[:, :K + 1].shape).astype(np.float32)
        Xc[:, :K + 1] = np.clip(Xc[:, :K + 1] + noise, 0.0, FMAX)
    return Xc


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    base = load_baseline(); mt, tj = train_tsjepa()
    rng = np.random.default_rng(0)

    print(f"\n=== ratchet MAE on CLEAN future vs observation noise sigma (lower=better) ===")
    print(f"  {'sigma':>6} | {'baseline':>10} | {'TS-JEPA':>10} | degrades less")
    base0 = jepa0 = None
    for s in SIGMAS:
        Xcn = noised(Xn, s, rng)
        Xc = torch.tensor(Xcn)
        with torch.no_grad():
            rb = rollout_from(base, Xc, cv, ev, K).numpy()
            rj = tj.decode_forecast(mt.dec, mt.enc(Xc, cv, tj.obs_mask(Xc.shape[0], tj.K_EVAL)),
                                    Xc, ev, tj.K_EVAL).numpy()
        mb = mae_over(rb, Xn, K + 1, T, RATCHETS)      # score vs the CLEAN truth
        mj = mae_over(rj, Xn, K + 1, T, RATCHETS)
        if s == 0.0:
            base0, jepa0 = mb, mj
        db = mb - base0; dj = mj - jepa0               # degradation from clean
        less = "TS-JEPA" if dj < db else "baseline"
        print(f"  {s:>6.2f} | {mb:>10.4f} | {mj:>10.4f} | {less}   (deg: base +{db:.4f}, jepa +{dj:.4f})")
    print("\n  'degrades less' = smaller error increase from its own clean baseline as noise rises.")


if __name__ == "__main__":
    main()
