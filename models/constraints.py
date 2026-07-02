"""
The by-construction constraint head, shared by the baseline and the JEPA decoder so BOTH
models inherit the identical hard guarantee (no duplicated, drift-prone math).

Given the previous raw state and the network's raw outputs, it produces the next state so that:
  - F, D, P, M  never decrease            (next = prev + softplus(raw)  >= prev)
  - A, C, flare stay in [0,1]             (next = sigmoid(raw), free to rise or fall)
  - S           never decreases EXCEPT it may step down at an ERCP month (relief gated by is_ercp)
  - every field stays within its bounds   (clamp to [0, fmax])
regardless of the network weights. Constraints are a property of this parameterisation.

Optionally (couple_m=True) it also enforces a COUPLING by construction, not just a per-field
rule: M's increment is gated by prev F * prev C, so M can only accumulate as a hazard of
sustained F*C -- the brief's "interesting part". This moves M from a freely-learned monotone
field to a derived/structured one, the same principle as cirrhosis = g(F) (see derived.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from generator import N_FIELDS, S, A, C, F, M, FLARE

FREE = [A, C, FLARE]   # bounded, non-monotone fields


class ConstraintHead(nn.Module):
    def __init__(self, couple_m: bool = False):
        super().__init__()
        self.couple_m = couple_m
        self.F_idx, self.C_idx = F, C
        self.register_buffer("fmax", torch.tensor([1, 1, 1, 1, 1, 1, 2, 1], dtype=torch.float32))
        free_mask = torch.zeros(N_FIELDS, dtype=torch.bool); free_mask[FREE] = True
        self.register_buffer("free_mask", free_mask)
        s_onehot = torch.zeros(N_FIELDS); s_onehot[S] = 1.0
        self.register_buffer("s_onehot", s_onehot)
        m_onehot = torch.zeros(N_FIELDS); m_onehot[M] = 1.0
        self.register_buffer("m_onehot", m_onehot)

    def forward(self, raw_fields, raw_relief, prev_x, is_ercp):
        """raw_fields:[...,8]  raw_relief:[...]  prev_x:[...,8]  is_ercp:[...] -> next [...,8]."""
        inc = Fn.softplus(raw_fields)          # >= 0  -> monotone increment (F,D,P,M and S creep)
        if self.couple_m:
            # M rises only proportional to sustained F*C (>=0, so M stays monotone by construction)
            fc = (prev_x[..., self.F_idx] * prev_x[..., self.C_idx]).unsqueeze(-1)
            inc = inc * (1.0 - self.m_onehot + self.m_onehot * fc)
        val = torch.sigmoid(raw_fields)        # (0,1) -> free-field absolute value
        nxt = torch.where(self.free_mask, val, prev_x + inc)
        relief = Fn.softplus(raw_relief) * is_ercp                # >= 0, only on ERCP months
        nxt = nxt - relief.unsqueeze(-1) * self.s_onehot
        # Final hard clamp to [0, fmax] -> bounds are GUARANTEED (test_invariants.py), incl. S after
        # aggressive ERCP relief. Tradeoff (readout-ready): clamp has zero gradient at the floor, so if
        # S saturated at 0 often the relief gradient would die there. It doesn't here (S~0.1-0.5,
        # relief~0.17, rarely pins to 0); if it did, a softplus-bounded transition would keep gradients.
        return torch.minimum(torch.clamp(nxt, min=0.0), self.fmax)
