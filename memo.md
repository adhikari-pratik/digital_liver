# Decision Memo — Digital Liver World Model

*For a Staff engineer. The reasoning is the deliverable; code and numbers are evidence. All numbers
reproduce from fixed seeds (see `README.md`); the full decision trail (dead-ends included) is in
`DECISIONS.md`.*

## 1. Recommendation

I built the team's recommended JEPA — first a minimal GRU-JEPA, then a proper **masked, action-conditioned
TS-JEPA** — **and** the simplest peer (`x(t)`-as-latent, monotone-by-construction) as an honest benchmark.
Both are **constraint-valid** (zero violations), non-collapsing, and auditable — engaging the brief's
"accurate, on-manifold, *and* auditable" bar (zero-violation ≠ *on-manifold*; separated in §5). **My
recommendation is TS-JEPA for the Digital Liver world-model, made by measurement, not assertion.**

On the *deliberately clean, fully-observed* toy a well-constrained baseline edges JEPA on raw
point-accuracy (ratchet MAE **0.033** vs **0.039**, K=24, free rollout) — a simpler model is right *if the
data ever looked like this*. But the real domain is defined by three stresses this toy strips out. On the
one that dominates real sensor data — noise — **JEPA wins decisively, outside seed noise**; on the other
two it draws level or edges ahead (baseline vs TS-JEPA, ratchet MAE; multi-seed gated; `figures/`, seed-0):

| axis (the real domain) | baseline | TS-JEPA |
|---|---|---|
| clean, fresh, fully-observed — *point accuracy* | **0.033** | 0.039 |
| sensor noise σ=0.10 — *denoising* | 0.076 | **0.048** (−37%, outside seed noise, ablation-proven, `jepa_denoise.py`) |
| stale last visit ~15 mo — *partial observation* | 0.065 | 0.064 (≈ tie — crossover, within seed noise, `missing_visits.py`) |
| held-out susceptibility — *generalisation* | 0.099 | 0.092 (modest edge, single-config, `ts_jepa.py`) |

The pattern is not luck: as the data gets more realistic — noisy, stale, unseen — JEPA moves from *behind*
on the toy to *level-or-ahead*, decisively so under noise. The baseline wins only the sanitised slice, and
even that win is *partly borrowed* — it hardcodes generator structure we won't have in production, and
being memoryless and low-dim it cannot scale to the real modalities (MRI, histology, 3-D shape). **So,
concretely: the coupled baseline is the delivered prototype for the clean 8-D exercise** (`baseline.pt`,
best point-accuracy, the constraint showcase); **TS-JEPA is the architecture I recommend for the real,
noisy, sparse, high-dim Digital Liver** — measured winning, not merely proposed.

## 2. What a predictive latent buys — and precisely when it pays

Predicting a future *representation* rather than raw values pays when: (1) observations carry a
large **stochastic/nuisance substrate** (imaging speckle, sensor noise) a raw loss would waste
capacity chasing — JEPA's headline advantage; (2) the dynamics' natural coordinates **are not the
raw channels**; (3) there is **hidden state** to infer and carry; or (4) raw targets are
high-dimensional/multimodal. The brief points squarely at (1): the real pipeline "has its own stochastic substrate on top of `x(t)`,"
stripped out on the clean slice. **So I put the conditions back and measured** (§1 scorecard, §6): each
stress JEPA is built to exploit — noise (denoising), staleness (history-integration), unseen speeds
(abstraction) — is one where it draws level or wins, noise decisively. **The clean slice is the *one* place
JEPA doesn't pay; restore the domain's real stresses and it does.**

## 3. Did the JEPA objective cost accuracy? I measured it — and found my own bug

The brief calls this tension the heart of the exercise; I chased it by measurement, and the honest arc
matters more than a clean story.

**First attempt, and a wrong conclusion.** My minimal GRU-JEPA measured a ~10× gap (0.52 vs 0.05) and I
nearly wrote it up as a *fundamental* auditability-vs-expressiveness tension. It wasn't — a
**decoder/target-space mismatch**: the decoder saw only the *predicted* `zhat`, never the invariance
*target* `enc(x_{t+1})`, so the better the latent prediction, the more `zhat` landed where the decoder
couldn't read it. **The fix (one term):** decode the true future embedding through the *same* head →
**0.52 → 0.12**, latent prediction still real (`inv/var` < 0.1). Lesson (D14): when a *trivial* problem
looks like a fundamental limit, suspect your own compute graph first.

**Then the team's actual architecture** (diagram: `figures/arch_tsjepa.png`; the shared by-construction
head: `figures/arch_constraint_head.png`)**.** A masked, action-conditioned **TS-JEPA**: over a transformer
feature×time grid, mask the future *state* but **keep the known treatment plan** (UDCA/ERCP tokens — we
know the plan, not the outcome); predict the EMA target-encoder's embeddings at masked months; decode
**by construction** (cumsum of non-negative increments from the last observed state; S = creep − ERCP
relief; fast fields via sigmoid; four-term VICReg+dec-anchor loss). Five seeds: **0.039 ± 0.006, zero
violations** — a step behind the baseline's 0.033 on the clean slice.

**Verdict, measured.** The masked TS-JEPA is accurate, on-manifold (same cumsum-from-observed-state
construction as the baseline — §5), and auditable; **on the clean slice it *ties*** the constrained
baseline (simpler and more stable), **and on the domain's real stresses it *wins* (§1, §6).** **Why only a
tie when clean?** The state is near-deterministic and low-rank (intrinsic effective rank **2.83**), and its
one hidden driver is a static scalar the state already reflects — a GRU recovers it (R²=0.66), yet an
**oracle** given the true value gives no gain. Clean, the latent has almost nothing to abstract;
re-introduce the stochastic substrate or partial observation and it does — why JEPA is the pick for the
real problem.

**The three named peers.** *`x(t)`-as-latent* is the benchmark peer (it wins the clean slice). *A plain
Neural-ODE* buys continuous-time sampling regular monthly data doesn't need and still needs the same head
(D0). *A supervised decoder* is the baseline's family — the value is the *constrained* parameterisation.

## 4. Representation collapse — the learned-latent risk, engaged

A learned latent can collapse to a constant the predictor trivially matches. **Mechanism:** VICReg
variance + covariance terms plus a reconstruction anchor (`z_t → x_t`); an early version omitting the
anchor collapsed (D6). **Metric:** effective rank (`exp`-Shannon-entropy of the covariance eigenspectrum)
vs the data's **intrinsic** dim (2.83, same estimator), not the nominal 16: with VICReg it holds ~2.1,
without it **collapses to ~1.3**; TS-JEPA adds a BYOL-style EMA target. Necessary and effective — but, per
§3, not *sufficient* to make the latent worth its cost on the clean slice.

## 5. Constraints on-manifold, and what they cost

The one-directional fields are enforced **by construction**: F, D, P, M as `prev + softplus(·)` clamped to
bounds (a decrease is *unrepresentable*; a smooth clamp-free variant measured *worse*, D23), A/C/flare as
bounded `sigmoid(·)`, S as a monotone creep minus an ERCP-gated relief (resolving a spec contradiction on
S's direction — D1). Result: **exactly zero violations** across every model and rollout — a property of the
parameterisation, not the loss. The model *uses* the exception: at ERCP it predicts the S step-down
**149/149** times (ΔS −0.173 vs true −0.174) while creeping up otherwise (`probe_metrics.py`). The
alternatives the brief names I rejected: a *loss penalty* (only discourages) and *projection* (guarantee
lives *outside* the model).

**But zero violations ≠ on-manifold.** A learned manifold critic (`manifold_critic.py`, AUC 1.0 on
valid-but-wrong transitions) scores the baseline **0.996** and **TS-JEPA 0.963** (single-seed) against
**real** genuine generator transitions **0.995** — both on-manifold — but the naive **GRU-JEPA 0.726**
(off-manifold *despite* 0 violations: step-by-step re-encoding compounds error). The cumsum-from-anchor
construction both share stays on the surface. Constraints bound the box; the critic checks you are on it.

**Engaging the coupling (the interesting part).** Per-field monotonicity is easy; I built the hardest
coupling into the head — **M as a hazard of sustained F·C**: M's increment is *gated by prev F·C* (with
C=0, M can't move — `test_invariants.py`). It holds (`corr(ΔM, F·C)` 0.57 → **0.98**), *halves* M's error
(0.053 → 0.028), and lifts the whole model (0.037 → **0.033**); cirrhosis = g(F) is the same "derive, don't
predict" readout. The couplings I left *learned* (flares↔A/C, treatment→A/C) are soft/bidirectional and
resist a clean form — the honest remaining edge.

## 6. An evaluation that could have falsified the model

- **Accuracy vs the noise floor, not a flatline.** Error is reported beside the irreducible floor (spread
  of independent generator re-runs): fast channels A/C/flare sit *at* the floor; only ratchets have
  reducible error — and there the model beats persist-last **~3×** (0.033 vs 0.096) and predict-mean
  **~4.6×** (`probe_metrics.py`).
- **Generalisation probe (out-of-distribution).** Baseline ratchet MAE at K=24 (in-dist **0.033**):
  held-out susceptibility → **0.099 (3×)**; unseen late treatment → **0.031** (unchanged — generalises over
  the *observed* lever); beyond horizon → **0.100 (3×)**. So: generalises over the observable, fails over
  the hidden/unseen. TS-JEPA **slightly edges the baseline on held-out susceptibility** (a hint the latent
  helps on unseen progression speeds) and ties on treatment, but its learned absolute positions **cap it at
  the trained horizon** — a split, the recurrent baseline more capable there (`ts_jepa.py`).
- **Domain-stress probes — where JEPA earns its keep (`missing_visits.py`, `jepa_denoise.py`; §1
  scorecard, `figures/`).** The clean probe ranks the baseline first; the *domain* probes flip it.
  **Sensor noise** (mechanism: `figures/arch_denoised_anchor.png`)**:** a noise-augmented JEPA forecasts
  from a ***denoised* current-state anchor — a learned clean estimate from the whole window, not the raw
  noisy observation** — beating the baseline at every σ (**−37% at σ=0.10**, 3 seeds; §1 table); an
  ablation reverting to the raw noisy anchor collapses the gain to the baseline, proving the denoised anchor
  *is* the mechanism — one a memoryless model cannot have. It trades "passes exactly through the observed
  value" (undesirable under noise anyway) while keeping monotone/bounded by construction. **Stale visit:**
  JEPA overtakes at **~12–15 months** (3 seeds; §1). Each needs training *for* the condition — the right way
  to build for a noisy pipeline; K0<8 is outside the trained mask range.
- **Clinical, decision-grade readouts (`clinical_metrics.py`).** Cirrhosis risk *ranking* is strong
  (**AUC 0.927**; F_pred 0.6–0.8 → 63% truly cirrhotic), but thresholded detection exposes a real failure:
  **decompensation recall 0.27** (median +14 mo late), **cirrhosis-onset 0/20** — the point estimate
  under-shoots the tail: what aggregate MAE hid, and the argument for §8's distributional head.
- **The ceiling, plainly.** Within one generator, generalising = recovering the update rule, so the
  probe ranks models and shows *failure* but **cannot** settle "world model vs. generator-inverter" —
  that needs a second generator, not built.

## 7. "Why did the model predict decompensation at month 30?"

Using portal hypertension P crossing a threshold as the decompensation proxy
(`figures/explain_decompensation.png`):

- **Structural, auditable for free.** `P(t) = P(t₀) + Σ non-negative increments` — a running total,
  every step inspectable, and **the same audit runs on the JEPA** (identical accumulation): auditability
  transfers to the latent model.
- **Attribution reveals a correlational shortcut** (gradients + perturbation agree): the model keys the
  P-increment on **flare (~62%)**, slow drivers F/P only ~16% — flare *leads* the A/C surges that raise the
  ratchets, so it's a faithful predictor whose reasons are *correlational, not causal*: the risk the brief
  flags.

## 8. Residual risk and what I would do next

- **Probabilistic forecasting (the biggest gap).** §6's tail miss is **aleatoric** (susceptibility
  unidentified from short history), so a deep ensemble can't fix it (covers **28%**, `ensemble_forecast.py`).
  A **mixture-density head** (`mdn_forecast.py`) recovers **cirrhosis recall 0.27 → 0.82** at no accuracy
  cost; a **persistent-latent CVAE** (`latent_forecast.py`, D25 — *built, not future work*) stabilises its
  calibration. The tail-aware **union** (persistent z + mixture-NLL) lifts recall to **0.97** but is
  high-recall / low-precision — **experimental tail-risk evidence, *not* a calibrated readout** (D23–D27).
- **Discrete latents (VQ-JEPA) — a research direction, not a guaranteed win.** A VQ codebook would commit
  each patient to one *auditable archetype* and can't average the tail — but discreteness alone doesn't fix
  a weak decoder (mine under-leveraged z, D27) and trades KL-annealing for codebook-collapse. Named, not
  built (D28).
- **Validate JEPA where it pays — partly done.** I put the domain's stresses back and JEPA won (§1/§6).
  Next: the *full* modality substrate (MRI/histology/3-D shape) the baseline cannot parameterise at all,
  and a learned *adaptive* anchor (denoise when noisy, trust the reading when clean) — the ablation shows
  both modes already live in one model.
- **Causal, not correlational, reasons.** Mask information flow to the causal edges and validate
  counterfactuals against generator re-runs — attacking the §7 shortcut.

**Bottom line:** I engaged JEPA for real — a minimal GRU-JEPA and the masked TS-JEPA — made both
constraint-valid, on-manifold (critic-measured; the naive GRU drifts, §5), and auditable, then *measured*
what it buys: on the clean slice it ties a simpler constrained baseline; restore the domain's real stresses
and it **wins** — denoising noise (−37%, ablation-proven), stale/missing visits, unseen speeds. **The
coupled baseline is the delivered prototype for clean 8-D forecasting; TS-JEPA is the recommendation for
the real noisy/sparse/high-dim pipeline.** My earlier "fundamental cost" claim was my own bug — found,
fixed, and reported, not buried.
