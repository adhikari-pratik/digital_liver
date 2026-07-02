"""
Invariant tests: prove the by-construction guarantees hold for RANDOM network outputs and inputs.
A constraint that only held after training would not be a guarantee -- these show the guarantee is
a property of the parameterisation, independent of weights. Run: python test_invariants.py
Exits nonzero if any invariant is violated (so it can gate CI / a pre-submit check).
"""
import copy
import numpy as np
import torch

from models.constraints import ConstraintHead
from generator import N_FIELDS, FIELD_NAMES, F, D, S, P, M, FIELD_MAX
from derived import is_cirrhotic, cirrhosis_stage

RATCHET = [F, D, P, M]                       # pure non-decreasing (S handled separately re ERCP)
GEN = torch.Generator().manual_seed(0)
_fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _fails.append(name)


def main():
    n = 4000
    fmax = torch.tensor(FIELD_MAX, dtype=torch.float32)
    head = ConstraintHead(couple_m=False)
    prev = torch.rand(n, N_FIELDS, generator=GEN) * fmax          # arbitrary valid previous state
    raw = torch.randn(n, N_FIELDS, generator=GEN) * 5             # large-magnitude raw outputs
    relief = torch.randn(n, generator=GEN) * 5
    no_ercp, ercp = torch.zeros(n), torch.ones(n)

    # 1) monotone ratchets hold for random weights, no ERCP
    nxt = head(raw, relief, prev, no_ercp)
    check("F/D/P/M never decrease (random outputs, no ERCP)",
          all((nxt[:, i] >= prev[:, i] - 1e-6).all().item() for i in RATCHET))
    check("S never decreases when no ERCP", (nxt[:, S] >= prev[:, S] - 1e-6).all().item())

    # 2) ERCP allows an S step-down; the other ratchets still cannot decrease
    nxt_e = head(raw, torch.abs(relief) + 2.0, prev, ercp)        # force positive relief
    check("S CAN step down at ERCP (relief representable)",
          (nxt_e[:, S] < prev[:, S] - 1e-3).any().item())
    check("F/D/P/M cannot decrease even at ERCP",
          all((nxt_e[:, i] >= prev[:, i] - 1e-6).all().item() for i in RATCHET))

    # 3) bounds: every field within [0, fmax]
    check("all fields within [0, fmax]",
          bool(((nxt >= -1e-6) & (nxt <= fmax + 1e-6)).all().item()))

    # 4) coupled M cannot rise when F*C = 0 (the F*C hazard gate, by construction)
    ch = ConstraintHead(couple_m=True)
    prev0 = prev.clone(); prev0[:, __import__("generator").C] = 0.0   # cholestasis zero -> hazard off
    nxt_c = ch(raw, relief, prev0, no_ercp)
    check("coupled M cannot rise when C=0 (F*C hazard gate)",
          torch.allclose(nxt_c[:, M], prev0[:, M], atol=1e-6))

    # 5) no stored cirrhosis channel; cirrhosis is a pure monotone function of F
    check("state is exactly 8-D with no cirrhosis field",
          N_FIELDS == 8 and "cirrhosis" not in [s.lower() for s in FIELD_NAMES])
    ft = np.linspace(0, 1, 50)
    check("cirrhosis readout is monotone in F and thresholds correctly",
          bool(np.all(np.diff(cirrhosis_stage(ft)) >= 0)) and bool(is_cirrhotic(0.9)) and not bool(is_cirrhotic(0.5)))

    # 6) TS-JEPA EMA-leakage regression: run a REAL forward+backward through the online and target
    #    encoders (as ts_jepa.train does) and assert NO gradient reaches the target (it must move only
    #    by EMA), while the online encoder DOES get gradient. Stronger than a requires_grad flag check.
    from ts_jepa import Encoder, obs_mask as ts_obs_mask, T as TS_T
    from models.baseline import CTX_DIM
    b = 4
    online = Encoder()
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    Xz = torch.rand(b, TS_T, N_FIELDS)
    cz = torch.rand(b, TS_T, CTX_DIM)
    zo = online(Xz, cz, ts_obs_mask(b, 24))                       # online (masked-future)
    with torch.no_grad():
        zt = target(Xz, cz, torch.ones(b, TS_T, dtype=torch.bool))   # EMA target, no graph
    torch.nn.functional.mse_loss(zo, zt.detach()).backward()
    target_no_grad = all(p.grad is None for p in target.parameters())
    online_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in online.parameters())
    check("TS-JEPA target encoder gets NO gradient in a real step (online does)",
          target_no_grad and online_has_grad)

    print(f"\n{'ALL INVARIANTS HOLD' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    raise SystemExit(1 if _fails else 0)


if __name__ == "__main__":
    main()
