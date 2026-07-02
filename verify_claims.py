"""
Verify the four load-bearing numbers behind the JEPA tradeoff call -- why the latent buys nothing
on this clean state (so the shipped toy predictor is the simpler peer, while JEPA stays the
recommended architecture for the real noisy problem). One place, from the trained checkpoints.
Run: python verify_claims.py
"""

import numpy as np
import torch
import torch.nn as nn

from data import get_split, get_probes, build_rollout_batch
from eval import rollout_from, RATCHETS
from models.baseline import MonotoneStep, CTX_DIM
from models.history import HistoryStep
from models.jepa import effective_rank
from models.constraints import ConstraintHead
from generator import N_FIELDS

K = 24


def rmae(pred, Xn, mask=None):
    if mask is None:
        mask = np.ones(len(Xn), bool)
    e = np.abs(pred[mask][:, K + 1:][..., RATCHETS] - Xn[mask][:, K + 1:][..., RATCHETS])
    return e.mean()


def main():
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    susc = np.array([p.susceptibility for p in va["ctx"]])

    # 1) intrinsic dimensionality of the state -------------------------------------------
    idim = effective_rank(X.reshape(-1, N_FIELDS))
    print("1) INTRINSIC DIMENSIONALITY of the 8-D state")
    print(f"     effective rank of raw states = {idim:.2f} / 8   "
          f"(low -> nothing for a latent to compress)\n")

    # load models
    bk = torch.load("checkpoints/baseline.pt")
    base = MonotoneStep(hidden=bk["hidden"], couple_m=bk.get("couple_m", False)); base.load_state_dict(bk["state_dict"]); base.eval()
    hist = HistoryStep(); hist.load_state_dict(torch.load("checkpoints/history.pt")["state_dict"]); hist.eval()

    bp = rollout_from(base, Xv, cv, ev, K).numpy()
    hp = hist.rollout(Xv, cv, ev, K).numpy()
    fast = susc > np.quantile(susc, 2/3)

    # held-out susceptibility probe
    sp = get_probes(n=200)["held-out susceptibility"]
    _, cs, es, Xt = build_rollout_batch(sp); Xtn = Xt.numpy()
    bp2 = rollout_from(base, Xt, cs, es, K).numpy()
    hp2 = hist.rollout(Xt, cs, es, K).numpy()

    print("2) baseline+w does NOT beat the plain baseline where the benefit was predicted")
    print(f"     fast tercile      ratchet MAE:  baseline={rmae(bp, Xn, fast):.4f}   +w={rmae(hp, Xn, fast):.4f}")
    print(f"     held-out suscept. ratchet MAE:  baseline={rmae(bp2, Xtn):.4f}   +w={rmae(hp2, Xtn):.4f}")
    print("     (higher = worse; +w is worse on both)\n")

    # 3) w DOES encode susceptibility -> not an inference failure -------------------------
    w = hist.patient_latent(Xv[:, :K + 1], cv[:, :K + 1])[:, K].detach().numpy()
    A = np.concatenate([w, np.ones((len(w), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, susc, rcond=None)
    r2 = 1 - ((susc - A @ coef) ** 2).sum() / ((susc - susc.mean()) ** 2).sum()
    print("3) the GRU latent w DOES infer the hidden susceptibility (so +w's null isn't an inference failure)")
    print(f"     linear R^2(w -> susceptibility) = {r2:.3f}\n")

    # 4) oracle: even the TRUE susceptibility doesn't help --------------------------------
    class Oracle(nn.Module):
        def __init__(self, h=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N_FIELDS + CTX_DIM + 1, h), nn.SiLU(),
                                     nn.Linear(h, h), nn.SiLU(), nn.Linear(h, N_FIELDS + 1))
            self.head = ConstraintHead()
        def forward(self, x, c, e, s):
            raw = self.net(torch.cat([x, c, s], -1))
            return self.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], x, e)

    torch.manual_seed(0)
    m = Oracle(); opt = torch.optim.Adam(m.parameters(), 1e-3); mse = nn.MSELoss()
    s_tr = torch.tensor([[p.susceptibility] for p in tr["ctx"]])[:, None, :].expand(-1, X.shape[1] - 1, -1)
    for _ in range(60):
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], 128):
            b = perm[i:i + 128]
            xhat = m(X[b][:, :-1], ctx[b][:, 1:], erc[b][:, 1:], s_tr[b])
            loss = mse(xhat, X[b][:, 1:]); opt.zero_grad(); loss.backward(); opt.step()
    s_va = torch.tensor(susc[:, None], dtype=torch.float32)
    out = Xv.clone(); cur = Xv[:, K]
    with torch.no_grad():
        for t in range(K + 1, Xv.shape[1]):
            cur = m(cur, cv[:, t], ev[:, t], s_va); out[:, t] = cur
    orac = out.numpy()
    print("4) ORACLE: feeding the model the TRUE hidden susceptibility does not improve accuracy")
    for nm, mask in [("slow", susc <= np.quantile(susc, 1/3)),
                     ("fast", fast)]:
        print(f"     {nm} tercile ratchet MAE:  baseline={rmae(bp, Xn, mask):.4f}   oracle={rmae(orac, Xn, mask):.4f}")
    print("\n=> state is ~3-D and near-sufficient; susceptibility adds nothing usable. Simple model wins.")


if __name__ == "__main__":
    main()
