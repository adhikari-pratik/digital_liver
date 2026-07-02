# Repository Audit Report

Role assumed: Principal AI/ML Research Scientist and Lead Technical Reviewer for a clinical world-model / digital-twin take-home.

Scope: audit of the current repository implementation, decision memo, assignment PDF, model graph, invariant tests, evaluation harnesses, and reproduced script outputs. This report reflects the current coupled-M baseline state.

## Executive Summary

The submission is now a strong Staff-level take-home artifact. The two major earlier issues have been addressed:

- The memo no longer conflates "zero constraint violations" with "on-manifold" in the opening; it now uses **constraint-valid** and separates manifold validity in section 5.
- The M<-F*C coupling is now actually shipped in `train.py` via `COUPLE_M = True`, and `checkpoints/baseline.pt` contains `couple_m: True`.

The current score is **4.6 / 5**. Remaining deductions are mostly presentation precision and stale ancillary documentation, not core graph correctness.

## 1. Gap Analysis: Code vs Decision Memo Alignment

The main implementation aligns well with the memo.

Hard constraints are implemented by construction in `models/constraints.py`:

- F, D, P, M use `prev + softplus(raw)`, so decreases are unrepresentable.
- A, C, and flare use bounded `sigmoid(raw)` values.
- S uses monotone creep minus `softplus(raw_relief) * is_ercp`, so downward S steps are gated on active ERCP.
- Bounds are enforced with a final clamp to `[0, fmax]`.
- With `couple_m=True`, M increments are multiplied by `prev_F * prev_C`, so M cannot rise when the sustained F*C hazard is absent.

The shipped baseline now enables the coupling:

```text
train.py: COUPLE_M = True
checkpoint: {'hidden': 64, 'couple_m': True}
```

`eval.py` and `compare.py` correctly reload this flag from the checkpoint.

Cirrhosis is correctly not stored or predicted. It is derived only from fibrosis in `derived.py` through `cirrhosis_stage(F)`, `is_cirrhotic(F)`, and `cirrhosis_onset_month(F_traj)`.

Remaining alignment caveat:

- `memo.md` says TS-JEPA is "on-manifold" because it uses the cumsum-from-observed-state construction. That is plausible and structurally defensible, but the manifold critic script currently scores real data, baseline, and GRU-JEPA, not TS-JEPA. A stricter phrasing would be "constraint-valid and structurally less drift-prone" unless a TS-JEPA critic score is added.

## 2. Silent Graph Bugs and Mathematical Leakage Audit

No future-observation leak was found in the main baseline, history model, or JEPA rollout paths. Baseline-style rollouts condition on true states through K and then autoregress.

TS-JEPA masking is structurally correct:

- Future state months are replaced by a mask token.
- Future treatment/action context is left visible, which is appropriate for known intervention plans.
- The target encoder is copied from the online encoder, frozen, evaluated under `torch.no_grad()`, and updated by EMA.

The new invariant test is strong. `test_invariants.py` now performs a real online/target encoder forward-backward pass and verifies that:

- the target encoder receives no gradients;
- the online encoder does receive gradients.

This closes the previous concern that target no-grad was only checked as a flag property.

`train_jepa.py` still intentionally backpropagates through the dec-anchor target embedding for the decode anchor. This is acceptable because it is the point of the D14 fix: the encoder target space is shaped to be decodable. The latent invariance term itself still uses `target.detach()`.

`jepa_sweep.py` now warns loudly at runtime that its old "fundamental tradeoff" conclusion is superseded and was caused by the decoder/target-space bug. It is acceptable as historical record.

## 3. Reproduced Verification Results

### Invariants

`python test_invariants.py`:

```text
[PASS] F/D/P/M never decrease (random outputs, no ERCP)
[PASS] S never decreases when no ERCP
[PASS] S CAN step down at ERCP (relief representable)
[PASS] F/D/P/M cannot decrease even at ERCP
[PASS] all fields within [0, fmax]
[PASS] coupled M cannot rise when C=0 (F*C hazard gate)
[PASS] state is exactly 8-D with no cirrhosis field
[PASS] cirrhosis readout is monotone in F and thresholds correctly
[PASS] TS-JEPA target encoder gets NO gradient in a real step (online does)
ALL INVARIANTS HOLD
```

### Baseline Evaluation

`python eval.py`:

```text
monotone/S violations: 0/58799
out-of-bounds: 0

K=12 ratchets: 0.0465
K=24 ratchets: 0.0325
K=36 ratchets: 0.0240

held-out susceptibility: 0.0990
unseen treatment timing: 0.0314
longer-than-training: 0.0336 in-horizon -> 0.1001 beyond month 60
```

The coupled baseline improves the old uncoupled K=24 ratchet MAE from roughly `0.0367` to `0.0325`.

### Flatline and ERCP Probe

`python probe_metrics.py`:

```text
model ratchets: 0.0325
persist-last ratchets: 0.0962
predict-mean ratchets: 0.1484

MVR F/D/P/M: 0.00%
ERCP drops caught: 149/149
mean predicted dS at ERCP drops: -0.1737
mean predicted dS at non-ERCP months: +0.0051
```

This rules out a flatline explanation and confirms the S action exception is being used, not merely permitted.

### Clinical Metrics

`python clinical_metrics.py`:

```text
decompensation recall: 12/45 = 0.27
median timing error: 14.5 months late
cirrhosis onset recall: 0/20 = 0.00
final cirrhotic rate: true 0.100 vs predicted 0.000
AUC using predicted final F as risk score: 0.927
```

The model ranks risk well but under-calls the high-risk tail. This remains the primary clinical failure.

### Head-to-Head

`python compare.py`:

```text
K=24 ratchet MAE:
baseline   0.0325
JEPA       0.1216
baseline+w 0.0625

held-out susceptibility:
baseline   0.0990
JEPA       0.1349
baseline+w 0.1856
```

The shipped coupled baseline remains the strongest model in the standard comparison.

### Coupling

`python coupling.py`:

```text
free-M      M MAE 0.0526, corr(dM,F*C) 0.574
structured  M MAE 0.0275, corr(dM,F*C) 0.978
structured-M monotone: True
```

This validates the shipped coupling change.

### Manifold Critic

`python manifold_critic.py`:

```text
critic AUC(real vs constraint-valid-corrupt): 1.000
real held-out:    0.995
baseline rollout: 0.996
GRU-JEPA rollout: 0.726
```

The baseline is on-manifold by this critic, while GRU-JEPA drifts despite zero hard-constraint violations.

## 4. Evaluation Honesty and Fault Lines

The evaluation is a clear strength. It includes:

- held-out accuracy;
- hard constraint violation rate;
- noise-floor comparison;
- naive flatline baselines;
- hidden susceptibility strata;
- held-out susceptibility;
- unseen treatment timing;
- beyond-training-horizon rollout;
- clinical event thresholds;
- manifold critic;
- ensemble uncertainty stress test.

The clinical failure is handled honestly. Aggregate MAE looks good, but event thresholds expose under-calling of decompensation and cirrhosis onset. The memo correctly frames this as a point-estimate limitation and a reason to implement a distributional/generative head.

The deep-ensemble argument remains directionally sound: model disagreement is too narrow to capture the high-risk tail. The current `ensemble_forecast.py` source now trains with `couple_m=True`, matching the shipped baseline. I did not rerun the full five-model ensemble in the latest pass because it retrains five models and is materially slower than the other checks.

## 5. Architectural Scorecard

Technical rigor and soundness: **4.4 / 5**

The core graph is clean, target-month context alignment is correct, checkpoint loading preserves structural flags, and the invariant test now exercises target-gradient behavior directly.

Structural safety and manifold invariants: **4.7 / 5**

Hard monotonicity, S gating, bounds, no stored cirrhosis, and M<-F*C gating are all enforced by construction and tested. The baseline also scores on-manifold by the learned critic.

Evaluation self-awareness: **4.8 / 5**

The evaluation actively falsifies the model and reports meaningful failures. The OOD probes and clinical threshold metrics are particularly strong.

Staff-level writing and communication: **4.4 / 5**

The memo is candid, technical, and decision-oriented. The main remaining writing risk is over-compressing "on-manifold" for TS-JEPA without directly scoring TS-JEPA in the manifold critic.

Overall: **4.6 / 5**

This is now a strong submission. The current deductions are mostly presentation precision and stale ancillary documentation.

## 6. Remaining Issues Before Submission

- `README.md` still says "JEPA is viable, accurate, on-manifold, and auditable" without the memo's corrected nuance. Update it to distinguish constraint validity, baseline/TS-JEPA cumsum construction, and GRU-JEPA manifold drift.
- `memo.md` says TS-JEPA is on-manifold by construction. This is defensible but stronger than what `manifold_critic.py` directly measures. Either add a TS-JEPA critic score or soften the wording.
- `AUDIT_REPORT.md` should be kept in sync with final numbers if additional changes are made.
- Historical numbers in `DECISIONS.md` are acceptable as a running log, but reviewers may skim. Keep superseded sections clearly marked.

## 7. Recommended Next Technical Step

The top clinical gap is tail under-calling. The right next architecture is a constrained distributional world model:

- Quantile head: emit multiple ordered future increment heads, train with pinball loss, and keep every quantile trajectory constraint-valid through `softplus` increments and ERCP-gated S relief.
- Mixture density / latent mixture head: infer mixture weights over progression modes from the history window, sample a mode, and decode each sampled trajectory through the existing constraint head.
- Susceptibility posterior: explicitly infer a distribution over hidden progression speed and roll out particles through the same constrained decoder.

The key principle: uncertainty should live in a latent progression posterior, while every sampled clinical trajectory remains biologically valid by construction.
