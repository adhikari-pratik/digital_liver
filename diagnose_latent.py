"""
DIAGNOSTIC: is the persistent-latent under-performance (recall 0.58, coverage 0.74) a BUG or a
genuine limitation? Trains one model (seed 0) and answers three questions:

  Q1  Does z ENCODE susceptibility?   corr(posterior mu_d, true susceptibility). If ~0 -> z is
      not capturing subtype (bug/under-training).
  Q2  Does the decoder USE z?         final-F when rolling out at posterior mean vs mu+2*std. If
      the spread is tiny -> the decoder ignores z (latent under-use), the real failure mode.
  Q3  Is the model's spread CALIBRATED to reality?  model predictive std(final F) vs the TRUE
      aleatoric std(final F) from re-running the generator per patient. If model << true, we are
      under-dispersed; the decomposition tells us whether it is z-posterior tightness or the
      within-trajectory flare noise a single fixed z structurally cannot represent.
"""
import numpy as np
import torch

from latent_forecast import PersistentLatent, train, K_OBS, S_SAMP, CIRRH
from data import get_split, build_rollout_batch, T
from generator import simulate, F as F_IDX


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("training persistent-latent (seed 0) for diagnosis...", flush=True)
    m, kld = train(seed=0, verbose=True)
    m.eval()
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    with torch.no_grad():
        mu, logvar = m.encode(Xv[:, :K_OBS + 1], cv[:, :K_OBS + 1])
    std = (0.5 * logvar).exp()
    susc = np.array([p.susceptibility for p in va["ctx"]])
    trueF = Xn[:, -1, 0]; true_cirr = trueF >= CIRRH

    # Q1: does z encode susceptibility?
    print(f"\nQ1  corr(posterior mu_d, true susceptibility)  (|corr| high => z captures subtype):")
    mun = mu.numpy()
    for d in range(mun.shape[1]):
        print(f"      z-dim {d}: corr = {corr(mun[:, d], susc):+.3f}   post_std(mean) = {float(std[:, d].mean()):.3f}")
    # best single-dim proxy
    best = max(range(mun.shape[1]), key=lambda d: abs(corr(mun[:, d], susc)))
    print(f"      best z-dim {best} |corr| = {abs(corr(mun[:, best], susc)):.3f}")

    # Q2: does the decoder USE z? roll out at mean vs mean + 2*std (the "fast" tail of the posterior)
    with torch.no_grad():
        f_mean = m.rollout(Xv, cv, ev, mu, K_OBS).numpy()[:, -1, F_IDX]
        # push z along +2 std in the susceptibility-aligned direction sign
        sgn = np.sign(corr(mun[:, best], susc)) or 1.0
        z_hi = mu.clone(); z_hi[:, best] = z_hi[:, best] + sgn * 2.0 * std[:, best]
        f_hi = m.rollout(Xv, cv, ev, z_hi, K_OBS).numpy()[:, -1, F_IDX]
    print(f"\nQ2  decoder z-sensitivity (final F): mean-z {f_mean.mean():.3f}  ->  +2std-z {f_hi.mean():.3f}"
          f"   (Delta {f_hi.mean() - f_mean.mean():+.3f}); if ~0, the decoder ignores z")

    # Q3: model predictive std vs TRUE aleatoric std of final F
    with torch.no_grad():
        Fs = np.stack([m.rollout(Xv, cv, ev, mu + torch.randn_like(std) * std, K_OBS).numpy()[:, -1, F_IDX]
                       for _ in range(S_SAMP)])          # [S,N]
    model_std = Fs.std(0)
    rng = np.random.default_rng(0)
    true_std = np.array([np.stack([simulate(p, T, rng)[-1, F_IDX] for _ in range(50)]).std()
                         for p in va["ctx"]])
    print(f"\nQ3  spread calibration of final F:")
    print(f"      model predictive std (z-induced) = {model_std.mean():.3f}")
    print(f"      TRUE aleatoric std (generator re-runs) = {true_std.mean():.3f}")
    ratio = model_std.mean() / max(true_std.mean(), 1e-6)
    print(f"      ratio model/true = {ratio:.2f}   (<1 => under-dispersed; how much is the gap)")

    # who are the true cirrhotics, and does the model's upper quantile reach them?
    q90 = np.quantile(Fs, 0.90, axis=0)
    print(f"\n    true cirrhotics (n={int(true_cirr.sum())}): mean true final F = {trueF[true_cirr].mean():.3f}")
    print(f"      their posterior-mean pred F = {f_mean[true_cirr].mean():.3f}, their q90 pred F = {q90[true_cirr].mean():.3f}")
    print(f"      (if q90 pred F < 0.8 for many, the upper quantile simply doesn't reach the tail)")


if __name__ == "__main__":
    main()
