# Decision Memo — Digital Liver World Model

*For a Staff engineer. The reasoning is the deliverable; code and numbers are evidence. All
numbers and figures reproduce from fixed seeds (see `README.md`); the full decision trail
(including dead-ends and one bug I found in my own work) is in `DECISIONS.md`.*

## 1. Recommendation

I built the simplest peer (`x(t)`-as-latent with a monotone-by-construction head) **and** the
team's recommended JEPA — first a minimal GRU-JEPA, then a proper **masked, action-conditioned
TS-JEPA**. All are **constraint-valid** (exactly zero violations), non-collapsing, and auditable —
engaging the brief's "accurate, on-manifold, *and* auditable" bar (zero-violation ≠ truly
*on-manifold*; I separate them in §5). Measured head-to-head, free rollout, ratchet MAE at K=24:

| model | ratchet MAE (K=24) | notes |
|---|---|---|
| constrained baseline **+ multistep, M←F·C coupled** | **0.033** (std ~0.001) | what I ship *here* |
| masked **TS-JEPA** (the team's direction) | **0.039 ± 0.006** (5 seeds) | competitive, 0 violations |
| GRU-JEPA, **dec-anchor fixed** | 0.12 | see §3 |
| GRU-JEPA, naive (my first attempt) | 0.52 | a *bug*, not a limit (§3) |

**On this deliberately clean toy the JEPA is viable and competitive but does not *beat* the
constrained baseline — which is also more stable (std 0.001 vs 0.006) and simpler — so I ship the
baseline here.** I commit to JEPA for the *real* Digital Liver, where §2's conditions hold, and I
built the masked architecture so that recommendation is *demonstrated, not deferred*. One honest
correction up front: my first read that JEPA carried a **fundamental** accuracy cost was wrong — it
was a bug in my own wiring, found and fixed (§3): "ties, doesn't beat," not "costs."

## 2. What a predictive latent buys — and precisely when it pays

Predicting a future *representation* rather than raw values pays when: (1) observations carry a
large **stochastic/nuisance substrate** (imaging speckle, sensor noise) a raw loss would waste
capacity chasing — JEPA's headline advantage; (2) the dynamics' natural coordinates **are not the
raw channels**; (3) there is **hidden state** to infer and carry; or (4) raw targets are
high-dimensional/multimodal. The brief points squarely at (1): the real pipeline "has its own
stochastic substrate on top of `x(t)`," stripped out here — sound for real data, but *this* clean
exercise is the one place it doesn't yet pay.

## 3. Did the JEPA objective cost accuracy? I measured it — and found my own bug

The brief calls this tension the heart of the exercise; I chased it by measurement, and the honest arc
matters more than a clean story.

**First attempt, and a wrong conclusion.** My minimal GRU-JEPA measured a ~10× gap (0.52 vs 0.05),
and I nearly wrote it up as a *fundamental* "auditability-vs-expressiveness" tension. It wasn't — it
was a **decoder / target-space mismatch**: the invariance loss drags `zhat` toward `enc(x_{t+1})`, but
the decoder is only ever trained on the *predicted* `zhat`, never on that target region. So the better
the latent prediction did its job, the more `zhat` landed in a space the decoder couldn't read, and
accuracy cratered *because* the JEPA objective was succeeding. Looks like a tension; was a wiring gap.

**The fix (one term).** Also decode the true future embedding through the *same* head and match it to
`x_{t+1}`, so the target becomes decodable. Result: **0.52 → 0.12**, a 4× gain, latent prediction
still real (`inv/var` < 0.1). Lesson (D14): when a result looks like a fundamental limit on a trivial
problem, suspect your own compute graph first.

**Then the team's actual architecture.** A masked, action-conditioned **TS-JEPA**: over a
transformer feature×time grid, mask the future *state* but **keep the known treatment plan** (UDCA/ERCP
tokens — we know the plan, not the outcome); predict the EMA target-encoder's embeddings at masked
months; decode **by construction** (cumsum of non-negative increments from the last observed state →
ratchets provably non-decreasing; S = creep − ERCP relief; fast fields via sigmoid; loss = the
four-term VICReg+dec-anchor objective with a rec-heavy→invariance warmup). Five seeds: **0.039 ± 0.006,
zero violations** — competitive with, though a step behind, the baseline's 0.033.

**Verdict, measured.** The masked TS-JEPA here is accurate, on-manifold (it decodes by the same
cumsum-from-observed-state construction as the baseline — §5), and auditable; it *ties*, it does not
beat, the constrained baseline, which is more stable and simpler. **Why only a tie?** The state is
near-deterministic and low-rank (intrinsic effective rank **2.83**), and its one hidden driver is a
static scalar the state already reflects — a GRU recovers it (R²=0.66) yet an **oracle** given the true
value gives no gain. The latent has almost nothing to abstract; the property JEPA monetizes (§2) is
what the toy removed.

**The three named peers.** *`x(t)`-as-latent* is what I ship. *A plain Neural-ODE* buys
continuous-time sampling that regular monthly data doesn't need and still needs the same head (D0).
*A supervised decoder* is the baseline's family — the value is the *constrained* parameterisation.
Re-attach the substrate and condition (1) returns: the direct test of the team's direction (§8), and
why TS-JEPA is built, not promised.

## 4. Representation collapse — the learned-latent risk, engaged

A learned latent can collapse to a constant the predictor trivially matches. **Mechanism:** VICReg
variance + covariance terms plus a reconstruction anchor (`z_t → x_t`); an early version omitting the
anchor collapsed (D6). **Metric:** effective rank (`exp`-Shannon-entropy of the covariance eigenspectrum,
Roy–Vetterli) vs the data's **intrinsic** dim (2.83, same estimator on the raw state), not the nominal
16: with VICReg it holds ~2.1; the ablation without it **collapses to ~1.3**. TS-JEPA adds a
BYOL-style EMA target on top. Necessary and effective — but, per §3, not *sufficient* to make the
latent worth its cost here.

## 5. Constraints on-manifold, and what they cost

The one-directional fields are enforced **by construction**: the head parameterises F, D, P, M as
`prev + softplus(·)`, clamped to bounds (a decrease is *unrepresentable*; the clamp's only wart — a
dead gradient at the floor — never bites here, and a smooth clamp-free variant measured *worse*, D23),
A/C/flare as bounded `sigmoid(·)`, and S as a monotone creep minus an ERCP-gated relief (resolving a
spec contradiction on S's direction — D1).
Result: **exactly zero violations** across every model and rollout — a property of the
parameterisation, not the loss. And the model *uses* the action exception, not just permits it: at
ERCP events in a free rollout it predicts the S step-down **149/149** times (ΔS −0.173 vs true −0.174)
while creeping up otherwise (`probe_metrics.py`) — the mechanic is learned. The alternatives the brief
names I rejected: a *loss penalty* (only discourages) and *projection* (guarantee lives *outside* the
model).

**But zero violations ≠ on-manifold.** A learned manifold critic (`manifold_critic.py`, AUC 1.0
rejecting valid-but-wrong transitions) scores the baseline rollout on-manifold (0.996 ≈
real 0.995) but the **GRU-JEPA free-rollout at 0.726**, despite its own 0-violation rate. The drift
is specific to *step-by-step re-encoding* (errors compound through the encoder); the
cumsum-from-last-observed-state construction used by *both* the baseline and TS-JEPA stays on the
manifold. Constraints bound the box; the critic checks you are on the surface inside it.

**Engaging the coupling (the interesting part), and it ships.** Per-field monotonicity is easy; I also
structured the hardest coupling into the shipped head — **M as a hazard of sustained F·C**: M's
increment is *gated by prev F·C*, so it can't rise unless both are high (with C=0, M can't move —
`test_invariants.py`). It holds (`corr(ΔM, F·C)` 0.57 → **0.98**), *halves* M's rollout error
(0.053 → 0.028), and improves the whole model (0.037 → **0.033**), so it ships; cirrhosis = g(F) is
the same "derive, don't predict" readout (`derived.py`). The couplings I left *learned* (flares↔A/C,
treatment→A/C) are soft/bidirectional and don't admit a clean form — the honest remaining edge.

## 6. An evaluation that could have falsified the model

- **Accuracy vs the noise floor, and not a flatline.** Error is reported beside the irreducible error
  (spread of independent generator re-runs): the fast channels A/C/flare sit *at* the floor; only the
  ratchets have reducible error. And the ratchet number isn't a flatline artifact — the model beats
  persist-last **~3×** (0.033 vs 0.096) and predict-mean **~4.6×** on the ratchets (`probe_metrics.py`).
- **Generalisation probe (out-of-distribution).** Baseline ratchet MAE at K=24 (in-dist **0.033**):
  held-out susceptibility → **0.099 (3×)**; unseen late treatment → **0.031** (unchanged — generalises
  over the *observed* lever); beyond horizon → **0.100 (3×)**. Generalises over the observable, fails
  over the hidden/unseen. TS-JEPA on the same probes **slightly edges the baseline on held-out
  susceptibility** (≈0.09 vs 0.099 — a hint the latent generalises to unseen progression speeds) and
  ties on treatment, but its learned absolute positions **cap it at the trained horizon** — a split,
  the recurrent baseline more capable on horizon (`ts_jepa.py`).
- **Clinical, decision-grade readouts (`clinical_metrics.py`).** Cirrhosis risk *ranking* is strong
  (**AUC 0.927**; predicted-F bins stratify cleanly: F_pred 0.6–0.8 → 63% truly cirrhotic). But
  thresholded event detection exposes a real failure: **decompensation recall 0.27** (median +14
  months late), **cirrhosis-onset recall 0/20** — the point estimate under-shoots the tail. A
  **point-estimate limit** that aggregate MAE hid, and the strongest argument for §8's distributional
  head (now built and measured).
- **The ceiling, plainly.** Within one generator, generalising = recovering the update rule, so the
  probe ranks models and shows *failure* but **cannot** settle "world model vs. generator-inverter" —
  that needs a second generator, not built.

## 7. "Why did the model predict decompensation at month 30?"

Using portal hypertension P crossing a threshold as the decompensation proxy
(`figures/explain_decompensation.png`):

- **Structural, auditable for free.** `P(t) = P(t₀) + Σ non-negative increments` — a running total,
  every step inspectable. **The same audit runs on the JEPA** (`explain_decompensation_jepa.png`):
  identical accumulation, same flare-shortcut (81%) — auditability transfers to the latent model.
- **Attribution reveals a correlational shortcut** (gradients + perturbation agree): the model keys
  the P-increment on **flare (~62%)**, the true slow drivers F/P only ~16% each. Flare *leads* the A/C
  surges that raise the ratchets — so it is a faithful *predictor* whose reasons are *correlational,
  not causal*: the correlation-vs-mechanism risk the brief flags.

## 8. Residual risk and what I would do next

- **Probabilistic forecasting (the biggest gap now) — and I built the fix, not just named it.** §6's
  tail miss is structural: a point estimate regresses to the conservative middle. The cheap fix fails —
  a **5-model deep ensemble** does *not* recover the tail (mean *worse* than one model; 90% intervals
  cover only **28%** — `ensemble_forecast.py`) because the uncertainty is **aleatoric** (hidden
  susceptibility is unidentified from a short history), not epistemic. So I trained the real fix — a
  **mixture-density head** (`mdn_forecast.py`, `models/distributional_head.py`): K components each
  decoded through the *same* `ConstraintHead`, so every sampled future is constraint-valid (the sampler
  draws a valid *mode*, never Gaussian noise that could break a ratchet). Measured over 3 seeds it
  recovers what the ensemble couldn't — **cirrhosis recall 0.27 → 0.82** at the upper quantile — at **no
  accuracy cost** (0.028 vs 0.033). Honest caveat: interval *calibration* stays seed-variable (coverage
  0.70 ± 0.15) — the memoryless per-step sampler under-commits to a persistent "fast-progressor" branch,
  so a persistent latent (or explicit calibration) is the last step.
- **Validate JEPA where it pays.** Re-attach the real modality substrate; once observations carry
  un-forecastable high-dimensional detail, the latent should overtake raw-space prediction — the
  direct test of §2, and TS-JEPA is **already built** for it. (A first attempt with *separable*
  nuisance failed — the raw model ignored the noise dims; redesign uses *entangled* noise, D12.)
- **Causal, not correlational, reasons.** Mask information flow to the causal edges and validate
  counterfactuals against generator re-runs — attacking the §7 shortcut.

**Bottom line:** I engaged JEPA for real — a minimal GRU-JEPA and the team's masked TS-JEPA — made
both constraint-valid and auditable (TS-JEPA and baseline on-manifold by construction; the naive GRU
drifts, §5), and *measured* that TS-JEPA is competitive with but not beaten by the constrained baseline
on this clean toy that strips its advantage. So I ship the simpler baseline here and commit to JEPA for
the real problem — and my earlier "fundamental cost" claim was a bug in my own code, found, fixed, and
reported rather than buried.
