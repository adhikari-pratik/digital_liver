"""
The distributional readout for the aleatoric tail (memo §8). The hidden susceptibility is unidentified
from a short history, so the same input admits multiple futures; a point estimate regresses to the
conservative middle and under-calls the cirrhosis/decompensation tail (deep ensembles don't fix this --
D19).

The fix: replace the point head with a MIXTURE over next-step states, decoding EACH component through
the existing by-construction `ConstraintHead`. So every mixture component -- and every sampled
trajectory -- is still constraint-valid (ratchets non-decreasing, S/ERCP gated, bounds hold). The
uncertainty lives in the mixture; the hard guarantee is untouched. `sample()` draws a constraint-valid
MODE (never mu + sigma*noise, which could break a ratchet). Trained by mixture NLL.

This module is the reusable head; it is TRAINED and MEASURED end-to-end in `mdn_forecast.py` (D23):
3 seeds, cirrhosis recall 0.27 -> 0.82 at no accuracy cost, interval calibration still seed-variable.
The __main__ below is a standalone smoke test (imports and runs, constraint-valid components).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import N_FIELDS
from models.constraints import ConstraintHead


class DistributionalHead(nn.Module):
    def __init__(self, hidden: int, n_mix: int = 3, couple_m: bool = True):
        super().__init__()
        self.K = n_mix
        self.pi = nn.Linear(hidden, n_mix)                        # mixture logits (branches)
        self.raw = nn.Linear(hidden, n_mix * (N_FIELDS + 1))      # per-component raw fields + S-relief
        self.log_sigma = nn.Linear(hidden, n_mix * N_FIELDS)      # per-component observation scale
        self.head = ConstraintHead(couple_m=couple_m)            # SAME hard guarantee, per component

    def forward(self, h, prev_x, is_ercp):
        """h:[B,hidden]  prev_x:[B,8]  is_ercp:[B] -> log_pi:[B,K], mu:[B,K,8] (each valid), sigma:[B,K,8]."""
        B = h.shape[0]
        log_pi = F.log_softmax(self.pi(h), dim=-1)
        raw = self.raw(h).view(B, self.K, N_FIELDS + 1)
        sigma = self.log_sigma(h).view(B, self.K, N_FIELDS).exp().clamp(1e-3, 1.0)
        prev = prev_x.unsqueeze(1).expand(B, self.K, N_FIELDS)    # broadcast prev state to K components
        erc = is_ercp.unsqueeze(1).expand(B, self.K)
        mu = self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], prev, erc)   # [B,K,8] -- each constraint-valid
        return log_pi, mu, sigma

    def nll(self, log_pi, mu, sigma, target):
        """Gaussian-mixture negative log-likelihood of the true next state under the K components."""
        t = target.unsqueeze(1)                                   # [B,1,8]
        log_comp = (-0.5 * (((t - mu) / sigma) ** 2) - sigma.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(log_pi + log_comp, dim=-1).mean()

    @torch.no_grad()
    def sample(self, log_pi, mu):
        """Draw a branch per row; return its (already constraint-valid) next state -> valid sampled futures."""
        k = torch.distributions.Categorical(logits=log_pi).sample()
        return mu[torch.arange(mu.shape[0]), k]


if __name__ == "__main__":   # smoke: it is real, runnable code -- not pseudocode
    torch.manual_seed(0)
    B, H = 5, 32
    head = DistributionalHead(hidden=H)
    h = torch.randn(B, H)
    prev = torch.rand(B, N_FIELDS)
    ercp = torch.zeros(B)
    log_pi, mu, sigma = head(h, prev, ercp)
    tgt = torch.rand(B, N_FIELDS)
    # every mixture component must be a valid next state (ratchets >= prev)
    ratchet = [0, 1, 3, 6]
    ok = all((mu[:, :, i] >= prev[:, i:i + 1] - 1e-6).all().item() for i in ratchet)
    print(f"shapes: log_pi{tuple(log_pi.shape)} mu{tuple(mu.shape)} sigma{tuple(sigma.shape)}")
    print(f"all K mixture components constraint-valid (ratchets >= prev): {ok}")
    print(f"mixture NLL on random target = {head.nll(log_pi, mu, sigma, tgt).item():.3f}")
    print(f"one constraint-valid sampled next state: {head.sample(log_pi, mu)[0].numpy().round(3)}")
