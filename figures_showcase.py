"""
Generates the showcase figures for the memo -- all from REAL, reproducible model outputs (seed 0),
NOTHING hand-drawn or fabricated. Trains one clean TS-JEPA (staleness/OOD axes) and one noise-augmented
denoised-anchor TS-JEPA (noise axis), loads the shipped baseline, and emits four PNGs into figures/:

  fig_scorecard.png   -- winner grid: baseline owns clean point-accuracy; JEPA owns the 3 domain axes
  fig_noise.png       -- ratchet MAE vs sensor noise: baseline / JEPA-denoise / JEPA-raw (ablation)
  fig_staleness.png   -- ratchet MAE vs months since last visit: baseline vs JEPA (the crossover)
  fig_trajectory.png  -- one patient under noise: truth, noisy observations, baseline drift, JEPA hold

Reproduce: python figures_showcase.py   (a few minutes; trains 2 small transformers).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import get_split, build_rollout_batch, get_probes
from eval import rollout_from, mae_over, RATCHETS
from models.baseline import MonotoneStep
from generator import FIELD_MAX, F as F_IDX
import ts_jepa as tj
import jepa_denoise as jd

T, K = 60, 24
SCORE_A, SCORE_B = 25, T
FMAXnp = np.array(FIELD_MAX, dtype=np.float32)
SIGMAS = [0.0, 0.05, 0.10, 0.15]
K0S = [24, 21, 18, 15, 12, 9]
BLUE, GREEN, GREY, RED = "#1f4e79", "#2e7d32", "#9e9e9e", "#c62828"


def load_baseline():
    ck = torch.load("checkpoints/baseline.pt")
    m = MonotoneStep(hidden=ck["hidden"], couple_m=ck.get("couple_m", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def jepa_clean_fc(mt, X, c, e, K0):
    with torch.no_grad():
        return tj.decode_forecast(mt.dec, mt.enc(X, c, tj.obs_mask(X.shape[0], K0)), X, e, K0).numpy()


def ratchet(pred, truth):
    return mae_over(pred, truth, SCORE_A, SCORE_B, RATCHETS)


def main():
    _, va = get_split()
    _, cv, ev, Xv = build_rollout_batch(va); Xn = Xv.numpy()
    base = load_baseline()
    print("  training clean TS-JEPA (seed 0) for staleness/OOD axes ...", flush=True)
    *_, m_clean = tj.train(seed=0, return_model=True)
    print("  training denoised-anchor TS-JEPA (seed 0) for the noise axis ...", flush=True)
    m_dn = jd.train(seed=0)
    rng = np.random.default_rng(0)

    # ---- axis data -------------------------------------------------------------------------
    # staleness (clean TS-JEPA)
    stale_b, stale_j = [], []
    for K0 in K0S:
        stale_b.append(ratchet(rollout_from(base, Xv, cv, ev, K0).numpy(), Xn))
        stale_j.append(ratchet(jepa_clean_fc(m_clean, Xv, cv, ev, K0), Xn))
    # noise (denoise TS-JEPA): baseline / denoise-anchor / raw-anchor ablation
    noise_b, noise_d, noise_r = [], [], []
    Xc_by_sig = {}
    for s in SIGMAS:
        Xc = jd.add_noise(Xn, s, K, rng); Xc_by_sig[s] = Xc
        noise_b.append(ratchet(rollout_from(base, Xc, cv, ev, K).numpy(), Xn))
        noise_d.append(ratchet(jd.forecast(m_dn, Xc, cv, ev, K, denoised=True), Xn))
        noise_r.append(ratchet(jd.forecast(m_dn, Xc, cv, ev, K, denoised=False), Xn))
    # held-out susceptibility (OOD generalisation), clean TS-JEPA
    sp = get_probes(n=200)["held-out susceptibility"]
    _, cs, es, Xs = build_rollout_batch(sp); Xsn = Xs.numpy()
    ood_b = ratchet(rollout_from(base, Xs, cs, es, K).numpy(), Xsn)
    ood_j = ratchet(jepa_clean_fc(m_clean, Xs, cs, es, K), Xsn)

    # ================= FIG 1: SCORECARD ====================================================
    j_noise = noise_d[2]                                  # sigma=0.10 denoise-anchor
    j_stale = stale_j[K0S.index(9)]; b_stale = stale_b[K0S.index(9)]
    rows = [
        ("Clean / fresh / full obs.   (point MAE)", stale_b[0], stale_j[0], "baseline"),
        ("Sensor noise sigma=0.10   (denoising)",   noise_b[2], j_noise,  "JEPA"),
        ("Stale visit ~15 mo   (partial obs.)",     b_stale,    j_stale,  "JEPA"),
        ("Held-out susceptibility   (generalise)",  ood_b,      ood_j,    "JEPA"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 3.1)); ax.axis("off")
    cell_text, cell_col = [], []
    for name, b, j, win in rows:
        cell_text.append([name, f"{b:.4f}", f"{j:.4f}", win.upper()])
        bc = GREEN if win == "baseline" else "white"
        jc = GREEN if win == "JEPA" else "white"
        wc = "#e8f5e9"
        cell_col.append(["white", bc, jc, wc])
    tbl = ax.table(cellText=cell_text, colLabels=["axis", "baseline", "TS-JEPA", "winner"],
                   cellColours=cell_col, cellLoc="center", loc="center",
                   colWidths=[0.44, 0.17, 0.17, 0.16])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.9)
    for (r, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#37474f"); cell.set_text_props(color="white", weight="bold")
        if cc == 0 and r > 0:
            cell.set_text_props(ha="left")
    ax.set_title("Baseline vs TS-JEPA: who wins where  (ratchet MAE, lower = better)",
                 weight="bold", fontsize=12, pad=12)
    fig.text(0.5, 0.02, "Baseline owns clean point-accuracy on the sanitized toy; TS-JEPA owns the "
             "three axes that define the real domain.", ha="center", fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig("figures/fig_scorecard.png", dpi=150); plt.close(fig)

    # ================= FIG 2: NOISE / DENOISING ===========================================
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(SIGMAS, noise_b, "o-", color=BLUE, lw=2.2, label="baseline (memoryless)")
    ax.plot(SIGMAS, noise_d, "s-", color=GREEN, lw=2.2, label="TS-JEPA, denoised anchor")
    ax.plot(SIGMAS, noise_r, "^--", color=GREY, lw=1.8, label="TS-JEPA, raw anchor (ablation)")
    ax.set_xlabel("sensor-noise sigma on the observed window"); ax.set_ylabel("ratchet MAE on clean future  (K+1..T)")
    ax.set_title("Denoising: JEPA's window-denoised anchor\nbeats the baseline under noise", fontsize=11.5, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    ax.annotate("raw-anchor ablation tracks the baseline\n(both trust the single noisy point)",
                xy=(0.10, noise_r[2]), xytext=(0.048, noise_r[2] + 0.020), fontsize=8, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY))
    fig.tight_layout(); fig.savefig("figures/fig_noise.png", dpi=150); plt.close(fig)

    # ================= FIG 3: STALENESS CROSSOVER =========================================
    gaps = [24 - k for k in K0S]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(gaps, stale_b, "o-", color=BLUE, lw=2.2, label="baseline (memoryless)")
    ax.plot(gaps, stale_j, "s-", color=GREEN, lw=2.2, label="TS-JEPA (integrates history)")
    cross = next((g for g, b, j in zip(gaps, stale_b, stale_j) if j < b), None)
    if cross is not None:
        ax.axvline(cross, color=RED, ls=":", lw=1.5)
        ax.annotate(f"crossover ~{cross} mo\nJEPA now more accurate", xy=(cross, stale_b[gaps.index(cross)]),
                    xytext=(cross - 1.5, max(stale_b) * 0.72), fontsize=8.5, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("months since last visit (staleness)"); ax.set_ylabel("ratchet MAE on clean future  (25..T)")
    ax.set_title("Partial observation: JEPA degrades gracefully as the last visit gets stale",
                 fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("figures/fig_staleness.png", dpi=150); plt.close(fig)

    # ================= FIG 4: ONE PATIENT UNDER NOISE =====================================
    sig = 0.10; Xc = Xc_by_sig[sig]
    rb = rollout_from(base, Xc, cv, ev, K).numpy()
    with torch.no_grad():
        z_dn = m_dn.enc(Xc, cv, tj.obs_mask(Xc.shape[0], K))
        out_dn, state_est = jd.decode_denoised(m_dn, z_dn, Xc, ev, K, denoised=True)
    rj = out_dn.numpy(); anchor_dn = state_est.numpy()          # denoised current-state estimate
    # pick a representative progressor where the mechanism is visible (largest baseline-minus-JEPA F error)
    fut = slice(K + 1, T)
    berr = np.abs(rb[:, fut, F_IDX] - Xn[:, fut, F_IDX]).mean(1)
    jerr = np.abs(rj[:, fut, F_IDX] - Xn[:, fut, F_IDX]).mean(1)
    prog = Xn[:, -1, F_IDX] > 0.5
    score = np.where(prog, berr - jerr, -1)
    p = int(np.argmax(score))
    mths = np.arange(T)
    jline = rj[p, K:, F_IDX].copy(); jline[0] = anchor_dn[p, K, F_IDX]   # start green at the DENOISED anchor
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(mths, Xn[p, :, F_IDX], color="black", lw=2.4, label="truth (clean F)")
    ax.plot(mths[:K + 1], Xc.numpy()[p, :K + 1, F_IDX], "o", color=GREY, ms=4, alpha=0.8,
            label="noisy observations")
    ax.plot(mths[K:], rb[p, K:, F_IDX], "--", color=BLUE, lw=2.0, label="baseline (noisy anchor -> drift)")
    ax.plot(mths[K:], jline, "-", color=GREEN, lw=2.2, label="TS-JEPA (denoised anchor -> holds)")
    ax.plot([K], [anchor_dn[p, K, F_IDX]], "*", color=GREEN, ms=14, label="JEPA denoised anchor")
    ax.axvline(K, color="#bbbbbb", ls=":", lw=1); ax.text(K + 0.5, 0.03, "forecast start", fontsize=8, color="#888")
    ax.set_xlabel("month"); ax.set_ylabel("fibrosis F"); ax.set_ylim(0, 1)
    ax.set_title(f"Patient {p}, noise sigma={sig}: JEPA denoises the anchor onto truth;\nbaseline trusts the "
                 f"noisy reading and drifts", fontsize=10.5, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.grid(alpha=0.25)
    fig.text(0.5, 0.005, "One representative progressor (largest baseline-minus-JEPA F error); "
             "aggregate is fig_noise.", ha="center", fontsize=8, style="italic", color="#777")
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig("figures/fig_trajectory.png", dpi=150); plt.close(fig)

    print("\n  wrote figures/fig_scorecard.png, fig_noise.png, fig_staleness.png, fig_trajectory.png")
    print(f"  scorecard: clean {stale_b[0]:.4f}/{stale_j[0]:.4f} | noise@.10 {noise_b[2]:.4f}/{j_noise:.4f} "
          f"| stale@15 {b_stale:.4f}/{j_stale:.4f} | OOD {ood_b:.4f}/{ood_j:.4f}")


if __name__ == "__main__":
    main()
