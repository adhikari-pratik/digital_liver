# Digital Liver World Model — take-home

A focused, honest slice of the world-model problem: predict how an 8-D clinical liver state
`x(t)` evolves month to month, with **hard constraints enforced by construction** and an
evaluation designed to falsify the model.

**Read `memo.md` first** (the 3-page decision memo — the primary deliverable). `DECISIONS.md` is
the full reasoning trail including dead-ends — and one bug I found in my own JEPA (D14).

## TL;DR result

Measured head-to-head, free rollout, ratchet MAE at K=24 (all models **0 constraint violations**):

| model | ratchet MAE (K=24) |
|---|---|
| constrained baseline **+ multistep, M←F·C coupled** (shipped here) | **0.033** (std ~0.001) |
| masked **TS-JEPA** (the team's direction) | **0.039 ± 0.006** (5 seeds) |
| GRU-JEPA, dec-anchor fixed (D14) | 0.12 |
| GRU-JEPA, naive first attempt | 0.52 (a *bug*, not a limit) |

- **JEPA is viable, accurate, on-manifold, and auditable — it *ties*, it does not beat, the simpler
  constrained baseline** on this clean, ~3-D, near-deterministic toy, which is also more stable. So I
  ship the baseline here and commit to JEPA for the *real* (noisy, high-dim) problem — and built the
  masked TS-JEPA (`ts_jepa.py`) so that recommendation is demonstrated, not deferred.
- **Honest correction:** my first read was that JEPA carried a *fundamental* accuracy cost
  (`jepa_sweep.py`). That was a decoder/target-space wiring bug, found and fixed (D14) — the gap was
  0.52→0.12, and a proper masked TS-JEPA reached ~0.04. `jepa_sweep.py` is kept as the record of the
  mistaken conclusion and how it was caught, not as a live claim.
- **Constraint-violation rate = 0.000000** (0 / 58,799 steps) — a property of the parameterisation.

## Setup

```
pip install torch numpy matplotlib
```
CPU is fine (models are ~15k params). All scripts are deterministic (fixed seeds).

## Run

```bash
python test_invariants.py  # prove the by-construction guarantees hold for RANDOM weights (monotonicity, S-gating, bounds, F*C gate, no cirrhosis channel)
python generator.py        # generate data + self-check: constraints hold, fields in range
python train.py            # baseline (x-as-latent, monotone-by-construction), one-step + MULTISTEP -> checkpoints/baseline.pt
python ts_jepa.py          # the masked, action-conditioned TS-JEPA (5 seeds) + OOD probe vs baseline -- the team's direction
python train_jepa.py       # minimal GRU-JEPA + dec-anchor fix (D14) + no-VICReg ablation -> checkpoints/jepa.pt
python train_history.py    # train baseline+w (native-space + GRU history latent)        -> checkpoints/history.pt
python eval.py             # baseline: 0-violation, accuracy vs noise floor, K-sweep, probe
python compare.py          # head-to-head: baseline vs GRU-JEPA vs baseline+w
python jepa_sweep.py       # SUPERSEDED (see D14): the invariance-weight sweep that first (wrongly) looked like a fundamental tradeoff
python jepa_variants.py    # JEPA round 2: 5 variants (EMA/BYOL, decode-weighted, ...) ablation ladder
python coupling.py         # engage the coupling: M<-F*C by construction + cirrhosis=g(F) readout
python manifold_critic.py  # learned on-manifold critic: 0 violations != on-manifold (GRU-JEPA rollout drifts)
python verify_claims.py    # the 4 load-bearing numbers behind the ship-the-baseline call
python clinical_metrics.py # decision metrics: event-timing error, cirrhosis AUC/precision/recall, population fidelity, interval coverage
python probe_metrics.py    # is 0.033 actually good? flatline check vs naive, MVR=0, action-conditional ERCP stricture drop
python make_training_curves.py # REAL train vs held-out learning curves per epoch -> figures/training_curve_*.png (+ .npz raw arrays)
python ensemble_forecast.py # probabilistic forecasting: deep ensemble tested & RULED OUT (tail is aleatoric, needs a distributional head)
python mdn_forecast.py     # the §8 fix, TRAINED (3 seeds): mixture-density head recovers cirrhosis recall 0.27->0.82 at no accuracy cost (D23)
python latent_forecast.py  # tests the persistent-latent hypothesis (seq-VAE, free-bits): stabilises calibration but tail-biased (D25)
python diagnose_latent.py  # WHY it under-performs: z encodes susceptibility, spread calibrated -> the cause is MSE tail-bias, not under-dispersion (D27)
python union_forecast.py   # the tail-aware fix: persistent z + per-step mixture-NLL -> recall 0.58->0.97 at best accuracy, precision/coverage tradeoff (D27)
python eval_mdn.py         # FAST (~9s, no training): verify the MDN tail claim from checkpoints/mdn.pt (one seed; 3-seed aggregate in mdn_forecast.py)
python smooth_head_test.py # tested Codex's clamp-free head: 0 violations but 0.039>0.033 -> clamp form kept (D23)
python explain.py          # "why decompensation at month 30?" (baseline) -> figures/explain_decompensation.png
python explain_jepa.py     # same audit, on the JEPA itself -> figures/explain_decompensation_jepa.png
```

## Files

| file | role |
|---|---|
| `generator.py` | seeded synthetic generator (data + quality bar); field indices; constraints |
| `test_invariants.py` | random-weight invariant tests: the guarantees are a property of the parameterisation |
| `make_training_curves.py` | real per-epoch train vs held-out learning curves (figures + raw `.npz`), for the baseline and TS-JEPA |
| `models/constraints.py` | `ConstraintHead` — the by-construction guarantee, shared by all models |
| `models/baseline.py` | `MonotoneStep` — the shipped model (memoryless, x-as-latent) |
| `models/jepa.py` | `JEPA` — GRU latent-space predictor + VICReg + effective-rank metric |
| `models/distributional_head.py` | §8 fix: mixture head, each component constraint-valid (design); **trained & measured** in `mdn_forecast.py` |
| `mdn_forecast.py` | trains the distributional head + MC rollout; 3-seed tail-recall/calibration result (D23) |
| `latent_forecast.py` | persistent-latent sequential-VAE (free-bits); tests the §8 calibration hypothesis (D25) |
| `diagnose_latent.py` | diagnoses the persistent latent: z-encoding, decoder z-use, spread calibration → MSE tail-bias (D27) |
| `union_forecast.py` | persistent z + per-step mixture-NLL (tail-aware); the confirmed fix, with its honest tradeoff (D27) |
| `eval_mdn.py` | fast MDN verification from `checkpoints/mdn.pt` (no training; one seed vs the 3-seed aggregate) |
| `smooth_head_test.py` | tested a clamp-free constraint head — kept the shipped clamp form (D23) |
| `ts_jepa.py` | **masked, action-conditioned TS-JEPA** — the team's direction, built & measured (D16) |
| `models/history.py` | `HistoryStep` — baseline + GRU history latent `w` |
| `coupling.py`, `derived.py` | M←F·C coupling by construction; cirrhosis = g(F) derived readout |
| `jepa_sweep.py` | invariance-weight sweep — **superseded by D14** (the "tradeoff" was a bug), kept as record |
| `jepa_variants.py` | JEPA round-2 ablation ladder (EMA/BYOL, decode-weighted, larger latent) |
| `manifold_critic.py` | learned "is-this-on-the-manifold" critic (try-something-new #3) |
| `data.py` | deterministic train/val split + OOD probe cohorts + input/target alignment |
| `train*.py` | training scripts (baseline is one-step + multistep; JEPA teacher-forced + dec-anchor) |
| `eval.py`, `compare.py`, `verify_claims.py` | evaluation, head-to-head, claim verification |
| `clinical_metrics.py` | decision metrics: event-timing, cirrhosis AUC/precision/recall, population fidelity, coverage |
| `probe_metrics.py` | segmented/flatline check vs naive baselines, MVR, action-conditional ERCP stricture drop |
| `ensemble_forecast.py` | deep-ensemble probabilistic test — ruled out (tail is aleatoric); §8 evidence |
| `explain.py`, `explain_jepa.py` | explainability worked example (baseline; and on the JEPA itself) |
| `memo.md`, `DECISIONS.md` | the memo (primary) and the full decision log |
| `figures/` | trajectory + explainability plots |

## Scope (what is deliberately out)

Modality decoders (each modality is a function of `x(t)`), graph-attention encoder,
counterfactual/causal validation, and continuous-time Neural-ODE — all discussed in `memo.md`
§8 / `DECISIONS.md` as next steps, not built. Ruthless scoping is part of the exercise.
