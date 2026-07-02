"""
!!! SUPERSEDED (see DECISIONS.md D14). The "no weight gives both accuracy and a real latent
prediction / fundamental tradeoff" conclusion below was an ARTIFACT of a decoder/target-space bug
(the decoder was never trained to decode the invariance target). Fixing that one term collapsed the
gap 0.52 -> 0.12, and a proper masked TS-JEPA (ts_jepa.py) reached 0.041, competitive with the
baseline. This script is kept ONLY as a record of the mistaken conclusion and how it was caught; do
NOT cite its "fundamental tension" framing. Run train_jepa.py (dec-anchor) and ts_jepa.py instead.

The load-bearing JEPA experiment: the auditability-vs-expressiveness tradeoff, measured.

The head-to-head shows the full JEPA (latent-prediction ON) at ~0.52 ratchet MAE vs the baseline's
0.052. Is that under-building, or a real property of the JEPA objective on this clean state? We
settle it by sweeping the weight on JEPA's DEFINING objective -- predicting the true future
embedding (latent-invariance) -- with a strong decode weight and VICReg on, and by ALSO measuring
whether that latent prediction is actually satisfied:

  inv/var = (invariance MSE) / (variance of the target embedding)
          ~0  -> zhat tracks the true future embedding: latent-prediction is REAL
          ~1  -> zhat is no better than predicting the mean: latent-prediction is OFF

Finding (see DECISIONS.md D13): there is NO weight that gives both accuracy AND a real latent
prediction. The better JEPA does its defining job, the worse its accuracy; accuracy returns only
when latent-prediction is switched fully off (at which point it is no longer JEPA). Monotone
across three orders of magnitude -> a genuine tradeoff, not a tuning fluke. Mechanism: the
by-construction head decodes raw, prev-relative increments (auditability lives in raw space);
latent-invariance wants an encoder in which dynamics are predictable forward; on this
near-deterministic state those are different latents and they fight over the shared encoder.

Run: python jepa_sweep.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_split, build_rollout_batch
from models.jepa import JEPA, variance_loss, covariance_loss, effective_rank
from eval import mae_over, RATCHETS
from generator import N_FIELDS

EPOCHS, BATCH, LR, SEED, K = 60, 128, 1e-3, 0, 24
DECODE = 5.0                              # fixed strong decode weight
INVS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BASELINE_REF = 0.0522                     # x-as-latent, K=24 ratchet MAE (from eval.py)


def train(w_inv):
    torch.manual_seed(SEED)
    tr, va = get_split()
    _, ctx, erc, X = build_rollout_batch(tr)
    _, cv, ev, Xv = build_rollout_batch(va)
    n = X.shape[0]
    m = JEPA()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    mse = torch.nn.MSELoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            Xb, cb, eb = X[b], ctx[b], erc[b]
            Hh = m.patient_latent(Xb, cb)
            z = m.enc(Xb[:, :-1]); w = Hh[:, :-1]
            zhat = m.pred(torch.cat([z, w, cb[:, 1:]], dim=-1))
            raw = m.dec(zhat)
            xhat = m.head(raw[..., :N_FIELDS], raw[..., N_FIELDS], Xb[:, :-1], eb[:, 1:])
            tgt = m.enc(Xb[:, 1:]).detach()
            recon = m.reconstruct(z)
            zf = z.reshape(-1, m.d_state)
            loss = (DECODE * mse(xhat, Xb[:, 1:]) + mse(recon, Xb[:, :-1])
                    + w_inv * mse(zhat, tgt) + variance_loss(zf) + covariance_loss(zf))
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        roll = m.rollout(Xv, cv, ev, K).numpy()
        z = m.enc(Xv[:, :-1]); w = m.patient_latent(Xv, cv)[:, :-1]
        zhat = m.pred(torch.cat([z, w, cv[:, 1:]], dim=-1))
        tgt = m.enc(Xv[:, 1:])
        inv_norm = float(((zhat - tgt) ** 2).mean()) / float(tgt.var().item() + 1e-9)
        er = effective_rank(z.reshape(-1, m.d_state))
    ratchet = mae_over(roll, Xv.numpy(), K + 1, Xv.shape[1], RATCHETS)
    return ratchet, er, inv_norm


def main():
    print("!!! SUPERSEDED (DECISIONS.md D14): the 'fundamental tradeoff' below was a decoder-anchor")
    print("    BUG, not a property of JEPA. Fixed -> 0.52->0.12; TS-JEPA -> 0.041. Kept as record.\n")
    print(f"baseline (x-as-latent) reference: K={K} ratchet MAE = {BASELINE_REF}")
    print(f"decode weight fixed at {DECODE}; VICReg on.\n")
    print(f"  {'inv weight':>10} | {'ratchet MAE':>11} | {'eff_rank':>8} | {'inv/var':>8} | latent-pred")
    print("  " + "-" * 62)
    maes, invnorms = [], []
    for wi in INVS:
        r, er, inv_norm = train(wi)
        maes.append(r); invnorms.append(inv_norm)
        state = "OFF" if inv_norm > 1 else ("weak" if inv_norm > 0.1 else "REAL")
        tag = "  <= competitive" if r < 0.065 else ""
        print(f"  {wi:>10.3f} | {r:>11.4f} | {er:>8.2f} | {inv_norm:>8.3f} | {state}{tag}")
    print(f"\n  => [SUPERSEDED conclusion, kept as record] No weight gives both low MAE and a REAL")
    print(f"     latent prediction (inv/var<0.1). At inv=1.0 latent prediction is ~perfect yet accuracy")
    print(f"     is {maes[-1]/BASELINE_REF:.1f}x the baseline; accuracy returns only at inv=0 (no longer JEPA).")
    print("\n" + "!" * 78)
    print("!!  DO NOT TRUST THE CONCLUSION ABOVE. It was a decoder/target-space BUG (DECISIONS D14),")
    print("!!  not a property of JEPA. Fixed -> gap 0.52->0.12; masked TS-JEPA (ts_jepa.py) -> ~0.04,")
    print("!!  competitive with the baseline. This script is a HISTORICAL RECORD of the mistake only.")
    print("!" * 78)

    # figure: the tradeoff curve
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(invnorms, maes, "o-", color="crimson")
    for wi, xn, y in zip(INVS, invnorms, maes):
        ax.annotate(f"inv={wi}", (xn, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(BASELINE_REF, ls="--", color="steelblue", label=f"x-as-latent baseline ({BASELINE_REF})")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("latent-prediction quality  (inv/var; left = REAL prediction, right = OFF)")
    ax.set_ylabel("ratchet MAE (K=24)")
    ax.set_title("Auditability vs expressiveness, measured:\nthe better JEPA predicts in latent, the worse its constrained accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    import os
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/jepa_tradeoff.png", dpi=110)
    print("\nsaved figures/jepa_tradeoff.png")


if __name__ == "__main__":
    main()
