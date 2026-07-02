"""
EXPERIMENT (branch exp/persistent-latent): test the memo's own §8 hypothesis.

The MDN head (mdn_forecast.py, D23) recovers the cirrhosis tail but its 90% interval is
UNDER-dispersed and seed-variable (coverage 0.70 +/- 0.15). Diagnosis (memo §8 / D23): its
sampler draws a fresh mixture MODE every step, so independent per-step draws random-walk back
toward the middle instead of COMMITTING to a persistent "fast progressor" trajectory. The memo
*predicts* the fix is a latent sampled ONCE per trajectory. This script measures that prediction.

Model (sequential VAE / CVAE over trajectories):
  - inference net: a GRU over the OBSERVED window x[0..K] (+context) -> posterior q(z | history),
    z in R^dz a per-patient "disease subtype" (susceptibility/branch).
  - transition: (x_t, ctx_t, z) -> raw -> the SAME by-construction ConstraintHead. z is FIXED for
    the whole rollout -> every sampled trajectory COMMITS to one subtype (the key difference vs MDN).
  - training: encode history -> sample z (reparam) -> free-roll K+1..T conditioned on z; loss =
    rollout MSE + beta * KL(q(z|hist) || N(0,I)), beta annealed (guard posterior collapse).
  - inference: draw S z's from the per-patient posterior q(z|history); each gives a committed
    trajectory -> predictive distribution over final F. Posterior spread = predictive spread.

Honest prior (before running): a persistent latent SHOULD widen coverage vs the memoryless MDN
(0.70 -> closer to 0.90). Risk: posterior collapse (KL -> 0, z ignored) would make it behave like
the memoryless model and NOT improve coverage -- an honest negative result if so. Watch the KL.
"""
import numpy as np
import torch
import torch.nn as nn

from data import get_split, build_rollout_batch, T
from generator import N_FIELDS
from models.constraints import ConstraintHead
from models.baseline import CTX_DIM
from eval import mae_over, RATCHETS

EPOCHS, BATCH, LR, HIDDEN, DZ = 80, 128, 1e-3, 64, 4
K_OBS, S_SAMP, CIRRH = 24, 300, 0.8
BETA_MAX = 0.5          # KL weight ceiling (annealed in); low enough to resist posterior collapse


class PersistentLatent(nn.Module):
    def __init__(self, ctx_dim=CTX_DIM, hidden=HIDDEN, dz=DZ, couple_m=True):
        super().__init__()
        self.dz = dz
        self.infer = nn.GRU(N_FIELDS + ctx_dim, hidden, batch_first=True)   # over observed window
        self.to_z = nn.Linear(hidden, 2 * dz)                               # -> mu, logvar
        self.trans = nn.Sequential(
            nn.Linear(N_FIELDS + ctx_dim + dz, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N_FIELDS + 1),
        )
        self.head = ConstraintHead(couple_m=couple_m)

    def encode(self, Xobs, ctxobs):
        """q(z | observed window 0..K): GRU over the window -> posterior mu, logvar."""
        h, _ = self.infer(torch.cat([Xobs, ctxobs], dim=-1))
        mu, logvar = self.to_z(h[:, -1]).chunk(2, dim=-1)
        return mu, logvar.clamp(-6, 4)

    def step(self, x, ctx, is_ercp, z):
        raw = self.trans(torch.cat([x, ctx, z], dim=-1))
        return self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], x, is_ercp)

    def rollout(self, X, ctx, ercp, z, K):
        out = X.clone(); cur = X[:, K]
        for t in range(K + 1, T):
            cur = self.step(cur, ctx[:, t], ercp[:, t], z); out[:, t] = cur
        return out


FREE_BITS = 0.5        # nats/dim the latent may use penalty-free -> prevents posterior collapse


def kl_freebits(mu, logvar, tau=FREE_BITS):
    """KL per dim with a free-bits floor: no pressure to push a dim below tau nats, so the
    encoder is not driven to the prior (the collapse failure of the first attempt)."""
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # [B, dz]
    return torch.clamp(kl_dim, min=tau).sum(-1).mean(), kl_dim.sum(-1).mean()


def onestep(m, Xb, cb, eb, z):
    """Teacher-forced one-step MSE over all months (stabilises training; pure-rollout is hard)."""
    zt = z.unsqueeze(1).expand(-1, Xb.shape[1] - 1, -1)
    raw = m.trans(torch.cat([Xb[:, :-1], cb[:, 1:], zt], dim=-1))
    nxt = m.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], Xb[:, :-1], eb[:, 1:])
    return ((nxt - Xb[:, 1:]) ** 2).mean()


def train(seed=0, verbose=False):
    torch.manual_seed(seed)
    tr, _ = get_split()
    _, ctx, erc, X = build_rollout_batch(tr); n = X.shape[0]
    m = PersistentLatent(); opt = torch.optim.Adam(m.parameters(), LR); mse = nn.MSELoss()
    last_kl = 0.0
    for ep in range(EPOCHS):
        beta = BETA_MAX * min(1.0, ep / (0.5 * EPOCHS))     # anneal KL in over first half
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]; Xb, cb, eb = X[b], ctx[b], erc[b]
            mu, logvar = m.encode(Xb[:, :K_OBS + 1], cb[:, :K_OBS + 1])
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()            # reparam
            pred = m.rollout(Xb, cb, eb, z, K_OBS)
            klv_pen, klv_true = kl_freebits(mu, logvar)
            loss = (mse(pred[:, K_OBS + 1:], Xb[:, K_OBS + 1:])             # committed free-rollout
                    + onestep(m, Xb, cb, eb, z)                             # teacher-forced stabiliser
                    + beta * klv_pen)
            opt.zero_grad(); loss.backward(); opt.step(); last_kl = float(klv_true.detach())
        if verbose and (ep % 20 == 0 or ep == EPOCHS - 1):
            print(f"  epoch {ep:3d}  KL/dim {last_kl / DZ:.3f}", flush=True)
    return m, last_kl


@torch.no_grad()
def evaluate(m, seed):
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    mu, logvar = m.encode(Xv[:, :K_OBS + 1], cv[:, :K_OBS + 1])
    std = (0.5 * logvar).exp()
    # accuracy: posterior-mean-z rollout
    pred_mean = m.rollout(Xv, cv, ev, mu, K_OBS).numpy()
    acc = mae_over(pred_mean, Xn, K_OBS + 1, T, RATCHETS)
    # predictive distribution: S samples from the per-patient posterior q(z|history)
    Ff = []
    for _ in range(S_SAMP):
        z = mu + torch.randn_like(mu) * std
        Ff.append(m.rollout(Xv, cv, ev, z, K_OBS).numpy()[:, -1, 0])
    Fsamp = np.stack(Ff)                                    # [S,N] final-F samples
    trueF = Xn[:, -1, 0]; true_cirr = trueF >= CIRRH
    q = lambda p: np.quantile(Fsamp, p, axis=0)
    lo, hi, q90 = q(0.05), q(0.95), q(0.90)

    def rp(flag):
        tp = int((flag & true_cirr).sum()); fp = int((flag & ~true_cirr).sum()); fn = int((~flag & true_cirr).sum())
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    p90, r90 = rp(q90 >= CIRRH)
    cov = float(((trueF >= lo) & (trueF <= hi)).mean())
    post_std = float(std.mean())
    return acc, r90, p90, cov, post_std


def main(seeds=(0, 1, 2)):
    print(f"persistent-latent (sequential VAE), {len(seeds)} seeds...", flush=True)
    rows = []
    for s in seeds:
        m, kld = train(s, verbose=(s == seeds[0]))
        acc, r90, p90, cov, pstd = evaluate(m, s)
        collapsed = kld / DZ < 0.02
        print(f"  seed {s}: acc={acc:.4f} q90 recall={r90:.2f}/prec={p90:.2f} coverage={cov:.2f} "
              f"post_std={pstd:.3f} KL/dim={kld/DZ:.3f}{'  <-- COLLAPSED' if collapsed else ''}", flush=True)
        rows.append([acc, r90, p90, cov])
    a = np.array(rows); mu, sd = a.mean(0), a.std(0)
    lbl = ["ratchet MAE", "q90 recall", "q90 prec", "90% coverage"]
    print(f"\n=== persistent-latent, {len(seeds)} seeds (mean +/- sd) ===")
    for i, name in enumerate(lbl):
        print(f"  {name:14} = {mu[i]:.3f} +/- {sd[i]:.3f}")
    print("  reference: MDN memoryless (D23) coverage 0.70+/-0.15, q90 recall 0.82; baseline MAE 0.033")
    print("  hypothesis: persistent z should push coverage 0.70 -> ~0.90 (if z does not collapse)")


if __name__ == "__main__":
    main()
