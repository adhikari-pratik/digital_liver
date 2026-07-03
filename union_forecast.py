"""
EXPERIMENT (branch exp/persistent-latent): the diagnostic (diagnose_latent.py) showed the
persistent-latent model is NOT under-dispersed (spread ratio 1.05 vs true aleatoric) and its z
DOES encode susceptibility (corr 0.5) -- but its MSE decoder is BIASED LOW on the tail (predicts
0.75 for cirrhotics who reach 0.92; regression-to-the-mean of a mean-seeking loss). So the fix is
not "more spread" -- it is a TAIL-AWARE objective.

Union model (the thing memo §8 predicted was the real fix): persistent latent z sampled ONCE per
trajectory (subtype) PLUS a per-step MIXTURE-density head conditioned on z (tail-aware via NLL, and
adds within-trajectory variation). Every mixture component decodes through the same ConstraintHead,
so every sampled future stays constraint-valid. Train by mixture-NLL over the rollout + KL(z).

Tests: does combining the two mechanisms lift q90 recall (0.58 persistent / 0.82 memoryless-MDN) and
coverage (0.74 / 0.70) toward nominal 0.90, WITHOUT losing the persistent latent's stability?
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from data import get_split, build_rollout_batch, T
from generator import N_FIELDS
from models.constraints import ConstraintHead
from models.baseline import CTX_DIM
from eval import mae_over, RATCHETS

EPOCHS, BATCH, LR, HIDDEN, DZ, NMIX = 80, 128, 1e-3, 64, 4, 3
K_OBS, S_SAMP, CIRRH = 24, 300, 0.8
BETA_MAX, FREE_BITS = 0.5, 0.5


class UnionModel(nn.Module):
    """Persistent latent z (subtype) + per-step mixture head (tail-aware, flare variation)."""

    def __init__(self, ctx_dim=CTX_DIM, hidden=HIDDEN, dz=DZ, n_mix=NMIX, couple_m=True):
        super().__init__()
        self.dz, self.K = dz, n_mix
        self.infer = nn.GRU(N_FIELDS + ctx_dim, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, 2 * dz)
        self.enc = nn.Sequential(nn.Linear(N_FIELDS + ctx_dim + dz, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, n_mix)
        self.raw = nn.Linear(hidden, n_mix * (N_FIELDS + 1))
        self.log_sigma = nn.Linear(hidden, n_mix * N_FIELDS)
        self.head = ConstraintHead(couple_m=couple_m)

    def encode_z(self, Xobs, ctxobs):
        h, _ = self.infer(torch.cat([Xobs, ctxobs], dim=-1))
        mu, logvar = self.to_z(h[:, -1]).chunk(2, dim=-1)
        return mu, logvar.clamp(-6, 4)

    def mixture(self, x, ctx, is_ercp, z):
        """One step: (x,ctx,z) -> mixture over next state. log_pi[B,K], mu[B,K,8], sigma[B,K,8]."""
        h = self.enc(torch.cat([x, ctx, z], dim=-1)); B = h.shape[0]
        log_pi = Fn.log_softmax(self.pi(h), dim=-1)
        raw = self.raw(h).view(B, self.K, N_FIELDS + 1)
        sigma = self.log_sigma(h).view(B, self.K, N_FIELDS).exp().clamp(1e-3, 1.0)
        prev = x.unsqueeze(1).expand(B, self.K, N_FIELDS)
        erc = is_ercp.unsqueeze(1).expand(B, self.K)
        mu = self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], prev, erc)   # each component valid
        return log_pi, mu, sigma

    def nll_step(self, log_pi, mu, sigma, target):
        t = target.unsqueeze(1)
        log_comp = (-0.5 * (((t - mu) / sigma) ** 2) - sigma.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(log_pi + log_comp, dim=-1)

    def mean_step(self, x, ctx, is_ercp, z):
        log_pi, mu, _ = self.mixture(x, ctx, is_ercp, z)
        return (log_pi.exp().unsqueeze(-1) * mu).sum(1)

    def sample_step(self, x, ctx, is_ercp, z):
        log_pi, mu, _ = self.mixture(x, ctx, is_ercp, z)
        k = torch.distributions.Categorical(logits=log_pi).sample()
        return mu[torch.arange(mu.shape[0]), k]


def kl_freebits(mu, logvar, tau=FREE_BITS):
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl_dim, min=tau).sum(-1).mean()


def train(seed=0, verbose=False):
    torch.manual_seed(seed)
    tr, _ = get_split(); _, ctx, erc, X = build_rollout_batch(tr); n = X.shape[0]
    m = UnionModel(); opt = torch.optim.Adam(m.parameters(), LR)
    for ep in range(EPOCHS):
        beta = BETA_MAX * min(1.0, ep / (0.5 * EPOCHS))
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]; Xb, cb, eb = X[b], ctx[b], erc[b]
            mu, logvar = m.encode_z(Xb[:, :K_OBS + 1], cb[:, :K_OBS + 1])
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            # teacher-forced per-step mixture NLL over the future, conditioned on the persistent z
            nll = 0.0
            for t in range(K_OBS + 1, T):
                lp, mm, sg = m.mixture(Xb[:, t - 1], cb[:, t], eb[:, t], z)
                nll = nll + m.nll_step(lp, mm, sg, Xb[:, t]).mean()
            loss = nll / (T - K_OBS - 1) + beta * kl_freebits(mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep % 20 == 0 or ep == EPOCHS - 1):
            print(f"  epoch {ep:3d}  nll {float(loss.detach()):.3f}", flush=True)
    return m


@torch.no_grad()
def rollout_mean(m, X, ctx, ercp, z, K):
    out = X.clone(); cur = X[:, K]
    for t in range(K + 1, T):
        cur = m.mean_step(cur, ctx[:, t], ercp[:, t], z); out[:, t] = cur
    return out


@torch.no_grad()
def rollout_sample(m, X, ctx, ercp, z, K):
    cur = X[:, K]
    for t in range(K + 1, T):
        cur = m.sample_step(cur, ctx[:, t], ercp[:, t], z)
    return cur[:, 0]                       # final-month F


@torch.no_grad()
def evaluate(m):
    _, va = get_split(); _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    mu, logvar = m.encode_z(Xv[:, :K_OBS + 1], cv[:, :K_OBS + 1]); std = (0.5 * logvar).exp()
    acc = mae_over(rollout_mean(m, Xv, cv, ev, mu, K_OBS).numpy(), Xn, K_OBS + 1, T, RATCHETS)
    # predictive dist: draw z once per sample (subtype) AND sample per-step modes (flare) -> both sources
    Ff = [rollout_sample(m, Xv, cv, ev, mu + torch.randn_like(std) * std, K_OBS).numpy() for _ in range(S_SAMP)]
    Fsamp = np.stack(Ff)
    trueF = Xn[:, -1, 0]; true_cirr = trueF >= CIRRH
    q = lambda p: np.quantile(Fsamp, p, axis=0)
    lo, hi, q90 = q(0.05), q(0.95), q(0.90)

    def rp(flag):
        tp = int((flag & true_cirr).sum()); fp = int((flag & ~true_cirr).sum()); fn = int((~flag & true_cirr).sum())
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    p90, r90 = rp(q90 >= CIRRH)
    cov = float(((trueF >= lo) & (trueF <= hi)).mean())
    return acc, r90, p90, cov


def main(seeds=(0, 1, 2)):
    print(f"union model (persistent z + per-step mixture-NLL), {len(seeds)} seeds...", flush=True)
    rows = []
    for s in seeds:
        m = train(s, verbose=(s == seeds[0]))
        acc, r90, p90, cov = evaluate(m)
        print(f"  seed {s}: acc={acc:.4f} q90 recall={r90:.2f}/prec={p90:.2f} coverage={cov:.2f}", flush=True)
        rows.append([acc, r90, p90, cov])
    a = np.array(rows); mu, sd = a.mean(0), a.std(0)
    for i, name in enumerate(["ratchet MAE", "q90 recall", "q90 prec", "90% coverage"]):
        print(f"  {name:14} = {mu[i]:.3f} +/- {sd[i]:.3f}")
    print("  refs: memoryless-MDN coverage 0.70+/-0.15 recall 0.82; persistent-z coverage 0.74+/-0.03 recall 0.58")


if __name__ == "__main__":
    main()
