"""
EXPERIMENT (branch exp/distributional-head): actually TRAIN the distributional head from
memo §8 -- not just sketch it -- and measure whether it recovers the aleatoric tail that the
point baseline (recall 0.27 on decompensation, 0/20 cirrhosis-onset) and the deep ensemble
(D19: spread too narrow, coverage 0.28) both miss.

Design under test:
  encoder (prev_x, ctx) -> h  ->  DistributionalHead: a K-component mixture over the next
  state, EACH component decoded through the SAME ConstraintHead, so every mixture component
  and every sampled trajectory is still constraint-valid. Train by one-step mixture NLL
  (the spread-teaching term) + a short multistep NLL on the mixture mean (the accuracy term).

Evaluated three ways, head-to-head with the shipped point baseline (0.033):
  1. accuracy   -- ratchet MAE of the mixture-mean rollout (does distributional cost accuracy?)
  2. tail recall-- MC-sample S trajectories; does an upper quantile of sampled final F catch
                   the cirrhosis tail the point estimate misses?
  3. calibration-- does the sampled 90% predictive interval actually cover true final F ~90%?

Honest prior (stated before running): a MEMORYLESS per-step mixture can spread but may not
COMMIT -- independent per-step sampling can random-walk back to the middle instead of
persisting a "fast progressor" draw. If so, the tail is still under-caught and the real fix
is a persistent latent, not just a mixture. Either outcome is a real finding. Measure it.
"""
import math
import numpy as np
import torch
import torch.nn as nn

from data import get_split, build_pairs, build_rollout_batch, T
from generator import N_FIELDS
from models.distributional_head import DistributionalHead
from eval import mae_over, RATCHETS
from models.baseline import CTX_DIM

EPOCHS, BATCH, LR, HIDDEN, NMIX = 80, 256, 1e-3, 64, 4
L = 6                      # multistep rollout length (matches the baseline's multistep)
K_OBS = 24                 # observe 0..K_OBS, free-roll the rest (matches head-to-head eval)
S_SAMP = 300               # MC trajectories per patient for the predictive distribution
CIRRH = 0.8                # final F >= 0.8 == cirrhosis tail


class MDNStep(nn.Module):
    """Encoder + mixture head: the trainable realisation of models/distributional_head.py."""

    def __init__(self, ctx_dim=CTX_DIM, hidden=HIDDEN, n_mix=NMIX, couple_m=True):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(N_FIELDS + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = DistributionalHead(hidden, n_mix=n_mix, couple_m=couple_m)

    def forward(self, x, ctx, is_ercp):
        h = self.enc(torch.cat([x, ctx], dim=-1))
        return self.head(h, x, is_ercp)                 # log_pi[B,K], mu[B,K,8], sigma[B,K,8]

    def mean(self, x, ctx, is_ercp):
        """Constraint-valid mixture mean = sum_k pi_k * mu_k (convex combo of valid states)."""
        log_pi, mu, _ = self.forward(x, ctx, is_ercp)
        return (log_pi.exp().unsqueeze(-1) * mu).sum(1)

    def sample_step(self, x, ctx, is_ercp):
        """Sample one component per row -> its (constraint-valid) next state."""
        log_pi, mu, _ = self.forward(x, ctx, is_ercp)
        k = torch.distributions.Categorical(logits=log_pi).sample()
        return mu[torch.arange(mu.shape[0]), k]


def train(seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, _ = get_split()
    x_in, ctx_tg, ercp_tg, x_tg = build_pairs(tr)            # one-step pairs
    _, cbig, ebig, Xbig = build_rollout_batch(tr)            # whole trajectories for multistep
    n = x_in.shape[0]; ntraj = Xbig.shape[0]
    model = MDNStep()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            log_pi, mu, sigma = model(x_in[b], ctx_tg[b], ercp_tg[b])
            loss = model.head.nll(log_pi, mu, sigma, x_tg[b])          # spread-teaching term
            # short multistep on the mixture MEAN -> keep the central path accurate under rollout
            tb = torch.randint(0, ntraj, (BATCH,)); s = int(rng.integers(0, T - 1 - L))
            cur = Xbig[tb, s]
            for k in range(L):
                lp, m2, sg = model(cur, cbig[tb, s + k + 1], ebig[tb, s + k + 1])
                loss = loss + model.head.nll(lp, m2, sg, Xbig[tb, s + k + 1]) / L
                cur = (lp.exp().unsqueeze(-1) * m2).sum(1)             # feed mean forward
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3d}  nll {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def mc_rollout(model, X, ctx, ercp, K, S):
    """Observe 0..K, then MC free-roll S sampled trajectories. Returns final F samples [S,N]."""
    N = X.shape[0]
    cur = X[:, K].repeat(S, 1)                        # [S*N, 8]
    csq = ctx.repeat(S, 1, 1); esq = ercp.repeat(S, 1)
    for t in range(K + 1, T):
        cur = model.sample_step(cur, csq[:, t], esq[:, t])
    return cur.view(S, N, N_FIELDS)[..., 0]           # final-month F, [S,N]


@torch.no_grad()
def mean_rollout(model, X, ctx, ercp, K):
    out = X.clone(); cur = X[:, K]
    for t in range(K + 1, T):
        cur = model.mean(cur, ctx[:, t], ercp[:, t]); out[:, t] = cur
    return out.numpy()


def run_seed(seed, Xv, cv, ev, Xn, verbose=True, save_path=None):
    """Train one MDN, return (acc, rec_q90, prec_q90, rec_q95, prec_q95, coverage)."""
    if verbose:
        print(f"\n--- seed {seed} ---", flush=True)
    model = train(seed); model.eval()
    if save_path:                       # persist this seed so eval_mdn.py can verify without training
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "n_mix": NMIX, "hidden": HIDDEN,
                    "couple_m": True, "seed": seed}, save_path)
        if verbose:
            print(f"  saved checkpoint -> {save_path}", flush=True)
    pred_mean = mean_rollout(model, Xv, cv, ev, K_OBS)
    acc = mae_over(pred_mean, Xn, K_OBS + 1, T, RATCHETS)
    Fsamp = mc_rollout(model, Xv, cv, ev, K_OBS, S_SAMP).numpy()
    trueF = Xn[:, -1, 0]; true_cirr = trueF >= CIRRH
    q = lambda p: np.quantile(Fsamp, p, axis=0)
    lo, hi, q90 = q(0.05), q(0.95), q(0.90)

    def rp(flag):
        tp = int((flag & true_cirr).sum()); fp = int((flag & ~true_cirr).sum()); fn = int((~flag & true_cirr).sum())
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    p90, r90 = rp(q90 >= CIRRH); p95, r95 = rp(hi >= CIRRH)
    cov = float(((trueF >= lo) & (trueF <= hi)).mean())
    if verbose:
        print(f"  acc={acc:.4f}  q90 recall={r90:.2f}/prec={p90:.2f}  q95 recall={r95:.2f}/prec={p95:.2f}  coverage={cov:.2f}")
    return acc, r90, p90, r95, p95, cov


def main(seeds=(0, 1, 2)):
    print(f"training MDN (mixture) head, {len(seeds)} seeds...", flush=True)
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    # save the FIRST seed's model so eval_mdn.py can verify the tail claim without retraining
    rows = np.array([run_seed(s, Xv, cv, ev, Xn,
                              save_path="checkpoints/mdn.pt" if i == 0 else None)
                     for i, s in enumerate(seeds)])
    m, sd = rows.mean(0), rows.std(0)
    lbl = ["ratchet MAE", "q90 recall", "q90 prec", "q95 recall", "q95 prec", "90% coverage"]
    print(f"\n=== MDN distributional head, {len(seeds)} seeds (mean +/- sd) ===")
    for i, name in enumerate(lbl):
        print(f"  {name:14} = {m[i]:.3f} +/- {sd[i]:.3f}")
    print("  reference: point baseline MAE 0.033, decompensation-tail recall 0.27; deep ensemble coverage 0.28 (D19)")


if __name__ == "__main__":
    main()
