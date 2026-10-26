# Design: PSM usefulness along one post-training trajectory (base → SFT → DPO → RLVR steps → final)

Draft, 2026-10-26. Builds on the `psm/decisions` experiment (`docs/psm/REPORT.md`) and the Q1 persona-landscape
pipeline. No grading manipulation: every measurement is in the ordinary, no-statement context.

## Question

**Is PSM still a useful framework for understanding model behaviour after intensive RLVR?** Concretely:

1. Can we quantitatively measure how a model's persona structure shifts over post-training?
2. Does PSM become a *stronger* framework after SFT, and a *weaker* one as RLVR accumulates?
3. North star (Neel): be able to say intelligent things about PSM in the high-RLVR world. We want statements of
   the form:
   * "persona-explained variance in behaviour goes X → Y over RLVR";
   * "Z% of what RL changed is persona-shaped rather than a uniform reflex".

## Why a trajectory, and why OLMo

* **Two endpoints can't show a trend.** Last night's OLMo DPO → final comparison found the persona landscape
  unchanged within noise. Persona's grip on decisions also held: stated values, discretionary actions, and
  irrelevance under explicit instructions. But OLMo's RLVR is mild.
* **The stronger-RL comparison is confounded.** Qwen3.6 → 3.8 does show a modest reorganisation:
  * landscape match 0.87/0.83 against a noise ceiling of 0.96–0.98;
  * slightly more one-dimensional;
  * the default Assistant's neighbours drift toward evaluator, reviewer and judge.

  But the releases differ in more than RL.
* **OLMo publishes the whole path, with the same tokenizer family throughout:**

| stage | checkpoint | HF repo / branch |
|---|---|---|
| pretrained base | Olmo-3-1125-32B | `allenai/Olmo-3-1125-32B` `main` |
| SFT (early) | Think-SFT, lr 5e-5, step 2000 of 10790 | `allenai/Olmo-3-32B-Think-SFT` `5e-5-step2000` |
| SFT (final) | Think-SFT | `allenai/Olmo-3-32B-Think-SFT` `main` |
| DPO | Think-DPO | `allenai/Olmo-3-32B-Think-DPO` (**done**) |
| RLVR steps | Olmo-3.1-32B-Think, 46 checkpoints, every 50 steps | `allenai/Olmo-3.1-32B-Think` `step_0050` … `step_2300` |
| RLVR final | Olmo-3.1-32B-Think | `main` (**done**) |

* **Proposed checkpoints:** base, SFT-early, SFT, DPO, RL 300, RL 800, RL 1500, RL final (2300+).
  * New runs: **6** (base, SFT-early, SFT, RL 300, RL 800, RL 1500).
  * DPO and final reuse last night's data wherever the measure is identical.
  * RL steps are spaced roughly geometrically; step 100 is already known to equal DPO.
* **Optional:**
  * the Qwen series 3.5 → 3.6 → 3.8 as a stronger-RL but confounded replication (Qwen3.5 is already pinned in
    the main checkout's `models.yaml`);
  * `allenai/Olmo-3-32B-Think` steps 50–750, a second, shorter RL run from the same DPO model.

## Measurements per checkpoint (all in the no-statement context)

### M1. Persona structure (Q1 pipeline, landscape only)

* **Prompts:** 50 OLMo roles + the default × 30 open-ended questions × prompt variant 0 × 2 samples, giving
  **3,060 generations**. Two samples allow noise-corrected statistics.
* **Activations:** last prompt token, answer-token mean, reasoning-token mean; layers 16–56.
* **Metrics** (the same estimators as `olmo_rl_compare.py` / `overlaps.py`):
  * representational similarity to DPO and to final, corrected for reliability with cross-half estimators;
  * participation ratio and PC1 share;
  * default-Assistant distance / persona spread, and its nearest neighbours;
  * how far PC1 aligns with the Assistant axis;
  * 50-way persona decoding accuracy on held-out questions (`docs/psm/decoding/decoding.py`);
  * role adherence (gpt-4.1-mini judge via OpenRouter, the Assistant Axis rubric).

### M2. Persona's grip on behaviour (PSM pipeline, 20 frozen personas, `data/psm/personas_v1.json`)

| behaviour family | items | per checkpoint |
|---|---|---|
| stated values | 72 dilemma items (6 dimensions), constrained, 2 samples | 2,880 |
| actions: discretionary vs instructed | reporting task, all 135 items incl. "It's your call" (suggest) and explicit-order items, 1 sample | 2,700 |
| stated vs action | questionnaire on the 72 conflict items | 1,440 |
| style / reflexes | 20 praise-inviting prompts (Olivia's 2d set) + length, exclamation and header markers from M1 answers | 400 |

**Metrics:**
* **Persona-explained variance** per family: between-persona variance over total item-level variance. It is
  noise-corrected using the sample-0 × sample-1 covariance (the estimator in `analysis.track_a`).
* **Stated → action predictiveness:** Spearman correlation across personas between dilemma honesty and reporting
  behaviour, and between questionnaire and report.
* **Landscape → behaviour:** Mantel ρ between persona activation distances and behaviour-profile distances.

### M3. Where post-training's changes live (persona-shaped vs reflex)

* For each behaviour *b* that moves between consecutive checkpoints, take the per-persona change
  Δ_r = y_r(t) − y_r(t−1). Fit Δ_r = c + β · z_r + ε, where z_r is the persona's Assistant-axis position at t−1.
* Report:
  * the **uniform share** c² / mean(Δ²): a reflex that applies to everyone;
  * the **Assistant-scaled share**: changes that fade with distance from the Assistant, i.e. "RL changed the
    Assistant character";
  * the **persona-specific residual**.

  This generalises the 10/24 update's 1c and 2c from two checkpoints to a curve.
* **Activation version:** for each persona centroid, the shift between checkpoints split into a shared shift +
  the part inside the previous checkpoint's persona subspace + the rest. This is valid because one trajectory
  keeps coordinates aligned (cos 0.96–1.00 measured).

### M4. Controllability (causal handle on persona)

* **Prompt controllability:** the M2 spreads, i.e. how much changing the system-prompt persona moves decisions.
* **Steering controllability** (default Assistant only):
  * directions: the Assistant axis and two persona offsets (comedian and contrarian, the extremes of
    discretionary falsification), plus 2 norm-matched random directions;
  * ±0.6 × A_L at layer 36;
  * outcomes: all 72 dilemma items + the 18 discretionary items × 2 samples.
  * That is 9 conditions × 108 = **~1,000 generations**.
* **Metric:** effect beyond the random band, per checkpoint. This asks whether persona directions still
  causally move decisions as RL accumulates.

### M5. Trait representations (cheap, forward passes only)

* Vignette honesty direction per checkpoint (`gshift.psm.vignettes`): held-out separation, and stability
  against DPO.
* Persona projections on it, and their correlation with M2 dilemma honesty.

## Base model

The base model has no chat template and no Assistant. Run it with a plain transcript format: role text, then
`User: …` / `Assistant:`, no thinking, stop at the next `User:`. M1 and M5 are fully comparable.

M2 for the base uses **log-prob forced choice**: compare P(" A") with P(" B") after `Choice:`, with no
sampling. This avoids format failures. Add the same log-prob readout for SFT and DPO, so that the base → SFT
step is measured with one instrument. SFT-early and SFT also show whether the Assistant persona appears early
or late in SFT.

## Pre-registered predictions (to make the result informative either way)

| | "PSM stays useful after RLVR" | "RLVR erodes PSM" |
|---|---|---|
| M1 landscape | RSA to DPO stays at the noise ceiling | RSA falls with RL steps; landscape compresses (lower participation ratio) |
| M2 persona-explained variance | flat or rising over RL steps, in every family | falls, first for actions, then for stated values |
| M2 stated → action | stable | falls |
| M3 change decomposition | RL changes mostly uniform reflexes or Assistant-scaled | RL changes persona-specific and unstructured |
| M4 steering | persona directions keep their effect | effects shrink toward the random band |

Prediction from last night's endpoints: the left column, with the largest movement at **base → SFT** (the
Assistant appears there) and little across RL. A falling trend across RL steps would be the newsworthy result.

## Cost and time (from last night's measured throughput)

**Throughput** (H200, thinking on, EasySteer):

| model | generations / s | tokens / generation |
|---|---|---|
| OLMo DPO | 3.0 | 1,000 |
| OLMo final | 2.0 | 1,490 |

| per new checkpoint | generations | GPU time |
|---|---|---|
| M1 landscape (generation + extraction) | 3,060 | ~35 min |
| M2 behaviour | 7,420 | ~50 min |
| M4 steering | ~1,000 | ~10 min |
| M5 + prompt activations (forward passes) | – | ~10 min |
| setup (weights, venv, server) | – | ~15 min |
| **total** | **~11,500** | **~2 h ≈ $11 on one H200** |

| item | estimate |
|---|---|
| 6 new checkpoints × ~$11 | ~$65 |
| contingency (retries, a slow pod, longer RL-step reasoning) ×1.5 | ~$100 total GPU |
| judges via OpenRouter (adherence ~18k short calls; flattery ~3k) | ~$15 |
| optional Qwen3.5 (M1 + M2) | +$15 |
| **total** | **~$115–130** (cap $250) |

**Wall-clock time:** 3 H200 pods × 2 checkpoints each gives ~4–5 h of compute.

| step | time |
|---|---|
| engineering | ~3–4 h |
| smoke test on one checkpoint | 30 min |
| runs | ~5 h |
| analysis + report | ~2–3 h |
| **total** | **≈ one working day** |

## What's reused vs new

| piece | status |
|---|---|
| Q1 generation / extraction / judge (`01_generate`, `02_extract`, `03_judge`, `experiment_olmo_rl.yaml`) | reuse; new stage "landscape-only" (baseline condition, 2 samples) |
| PSM assay, dilemma panel, questionnaire, scoring, pod runners (`run_psm*.sh`), tolerant I/O | reuse; new stage `traj` = base context only, Track A jobs + steering |
| decoding (`docs/psm/decoding/decoding.py`), overlaps, landscape metrics, `compare_rl.py` | reuse; generalise from pairs to sequences |
| vignettes / honesty direction | reuse as-is |
| `models.yaml` entries for 6 checkpoints (pinned revisions from branch SHAs) | new (small) |
| base-model transcript template + log-prob forced choice | **new** |
| M3 change decomposition (uniform / Assistant-scaled / residual) and trajectory figures | **new** |
| persona-direction steering (Assistant axis, persona offsets) | new directions; existing EasySteer path |

## Risks

* **The base model may not follow the transcript or the report format.** Mitigation: log-prob readouts, and
  base results reported as structure-only if validity is low.
* **OLMo's RLVR is mild next to frontier RL.** The trajectory gives a dose–response within it, but
  extrapolating to "intensive RLVR" is a stretch. That's what the Qwen series is for, with its confounds stated.
* **20 personas limit power for between-persona statistics.** M1 uses 50 roles; M2 could expand to 30 personas
  for about +50% cost.
* **The SFT-early checkpoint may be incoherent.** If it can't follow the format, drop it.
