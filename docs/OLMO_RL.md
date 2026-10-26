# How does RL change the persona landscape? (OLMo RL-stage comparison)

The Assistant Axis persona pipeline from this repo, run on three checkpoints of **one** RLVR
trajectory, so the only thing that changes between them is the amount of RL.

| Stage | Checkpoint | Revision | Model key |
|---|---|---|---|
| Before RLVR | `allenai/Olmo-3-32B-Think-DPO` | `97604023` (main) | `olmo3-32b-dpo` |
| A little RLVR (~4% of the run) | `allenai/Olmo-3.1-32B-Think` | `e7b80281` (branch `step_0100`) | `olmo31-32b-rl100` |
| Full RLVR (step 2,300+) | `allenai/Olmo-3.1-32B-Think` | `97ade57a` (main) | `olmo31-32b-final` |

`Olmo-3.1-32B-Think` is RLVR on Dolci-Think-RL started from `Olmo-3-32B-Think-DPO`, as its
model card states.

## Design

The design file is [`configs/experiment_olmo_rl.yaml`](../configs/experiment_olmo_rl.yaml).
It is the Qwen design v3 with these changes, applied identically to all three checkpoints.

**Prompts**
- 50 personas: the original 10 per category, with 40 discovery and 10 held out.
- Prompt variant 0 only.
- All 60 questions: 30 open-ended and 30 checkable (coding, arithmetic, structured data).

**Conditions.** Each condition is a sentence prefixed to the user message.
- Open-ended questions: baseline, automated grader, human grader, and a length-matched neutral sentence.
- Checkable questions: baseline, {automated, human} × {correctness, quality}, and a length-matched neutral sentence.
- Graded minus neutral separates the grading-specific effect from the effect of inserting any sentence.

**Generation and judging**
- 15,300 generations per checkpoint, with thinking on.
- OLMo's recommended sampling: temperature 0.6, top_p 0.95, no top-k, 12k-token budget.
- Role adherence is judged by gpt-4.1-mini with the Assistant Axis rubric.

**Activations.** Activations are taken from the last prompt token (`pre`) and as means over the
final-answer tokens (`resp`) and reasoning tokens (`reason`). Layers: 16–56 (64 blocks; layer 36
is primary, as in Qwen).

**Tokenizer.** All three checkpoints render prompts with the final checkpoint's tokenizer and
chat template. Vocabulary and merges are byte-identical; the DPO template differs only in its
default system prompt.

## Data (public)

Everything is on the HF dataset
[timf34/grader-shifts-persona-space-outputs](https://huggingface.co/datasets/timf34/grader-shifts-persona-space-outputs),
under `<model key>/`:

- `<condition>/responses/<role>.jsonl`: generations, with the reasoning/answer split
- `<condition>/activations/<role>/{pre,resp,reason}.npy`: [records, 7 layers, 5120]
- `<condition>/scores/<role>.jsonl`: role-adherence scores
- `analysis/summary.json`, `analysis/vectors/`, `analysis/tables/`: Q1 analysis (all layers and representations)
- `analysis/figures_report/`: figures

## First results

Layer 36, point estimates. Each cell gives shift size / leave-one-persona-out cosine / spread ratio:
- **Shift size:** the mean shift's norm relative to the typical activation norm.
- **Leave-one-persona-out cosine:** how much the personas agree on the shift's direction.
- **Spread ratio:** the spread of the personas under the condition relative to baseline. Values below 1 mean the personas converge.

| | DPO | step 100 | final |
|---|---|---|---|
| prompt token, open-ended, automated − neutral | 0.19 / 0.92 / 0.93 | 0.19 / 0.91 / 0.93 | 0.21 / 0.92 / 0.89 |
| prompt token, checkable, auto+correct − neutral | 0.26 / 0.98 / 0.81 | 0.26 / 0.97 / 0.81 | 0.28 / 0.97 / 0.76 |
| answer tokens, open-ended, automated − neutral | 0.016 / 0.44 / 1.26 | 0.017 / 0.44 / 1.65 | **0.053 / 0.76 / 1.11** |
| answer tokens, checkable, auto+correct − neutral | 0.095 / 0.92 / 0.67 | 0.086 / 0.90 / 0.46 | **0.143 / 0.92 / 0.42** |

Role adherence change on checkable tasks under automated + correctness grading:
−0.12 (DPO), −0.14 (step 100), −0.19 (final). The neutral-sentence control gives −0.00 to −0.04.

## Confirmed results (with 95% CIs)

Full report: [`olmo_rl_compare/report.md`](https://huggingface.co/datasets/timf34/grader-shifts-persona-space-outputs/blob/main/olmo_rl_compare/report.md)
(figures in `olmo_rl_compare/figures/`; code `scripts/olmo_rl_compare.py`). Layer 36, paired persona bootstrap; Δ = final − DPO.

- **A little RL (step 100) changes nothing.** It matches DPO on every measure.
- **The persona landscape keeps its shape.** Prompt-token distance matrices correlate 0.975 [0.94, 0.99] between DPO and final; spread, dimensionality and the default Assistant's neighbours (evaluator, advocate, scientist) are unchanged. Answer-token landscapes keep their arrangement (0.985) but spread out ~60%.
- **Behaviour under grading changes.** On open-ended questions, "evaluated automatically" lowers role adherence only after RL: DPO +0.04, final −0.23 (difference −0.27 [−0.39, −0.15]). The length-matched neutral sentence has no effect; human grading moves the same way, about half as much.
- **Answers get much shorter under automated grading:** 0.51× baseline length after RL vs 0.86× before.
- **The answer-token grading shift grows ~3× and rotates, but it mostly tracks answer shortening** (r ≈ −0.75 across personas). Shorter answers leave less room for role-play, so part of the adherence drop may be length.
- **The prompt-token grading direction barely changes:** ~8% larger, same direction (cos 0.98), slightly more persona convergence.

**Reading:** RL does not create or reorganise the grader representation; it changes what the model does with it.

## Steering across checkpoints

[`configs/steering_olmo.yaml`](../configs/steering_olmo.yaml): each checkpoint is steered along the **same** prompt-token grading directions (estimated on the DPO model, and as a check on the final model) at ±0.1 / ±0.3 × each model's mean residual norm, layer 36, against two random directions. Report: [`olmo_steer_compare/report.md`](https://huggingface.co/datasets/timf34/grader-shifts-persona-space-outputs/blob/main/olmo_steer_compare/report.md) (code `scripts/olmo_steer_compare.py`).

- **Steering does not reveal an RL-specific effect.** No grading direction changes role adherence, School of Reward Hacks or agentic misalignment beyond the random-direction band in any checkpoint; the final − DPO differences of effects are at chance level (2 of ~100 CIs exclude 0, with no pattern). Final-model directions behave like DPO ones.
- **RL itself increases reward hacking.** Unsteered, the final model's School of Reward Hacks gap (metric − quality, 0-100) is +5.1 [+2.5, +7.7] above DPO (metric +2.9 [+1.0, +4.9]); step 100 = DPO. Agentic harmful rate −0.05 [−0.13, +0.02]; adherence unchanged.
- **Reading:** the grading representation predates RL and pushing on it at these strengths barely moves behaviour; RL changes the behaviour downstream of it. Caveats: modest strengths at one layer (4× breaks the output), 150 agentic samples per condition, one sample per prompt.

## First results

Layer 36, point estimates. Each cell gives shift size / leave-one-persona-out cosine / spread ratio:
- **Shift size:** the mean shift's norm relative to the typical activation norm.
- **Leave-one-persona-out cosine:** how much the personas agree on the shift's direction.
- **Spread ratio:** the spread of the personas under the condition relative to baseline. Values below 1 mean the personas converge.

| | DPO | step 100 | final |
|---|---|---|---|
| prompt token, open-ended, automated − neutral | 0.19 / 0.92 / 0.93 | 0.19 / 0.91 / 0.93 | 0.21 / 0.92 / 0.89 |
| prompt token, checkable, auto+correct − neutral | 0.26 / 0.98 / 0.81 | 0.26 / 0.97 / 0.81 | 0.28 / 0.97 / 0.76 |
| answer tokens, open-ended, automated − neutral | 0.016 / 0.44 / 1.26 | 0.017 / 0.44 / 1.65 | **0.053 / 0.76 / 1.11** |
| answer tokens, checkable, auto+correct − neutral | 0.095 / 0.92 / 0.67 | 0.086 / 0.90 / 0.46 | **0.143 / 0.92 / 0.42** |

Role adherence change on checkable tasks under automated + correctness grading:
−0.12 (DPO), −0.14 (step 100), −0.19 (final). The neutral-sentence control gives −0.00 to −0.04.

## Confirmed results (with 95% CIs)

Full report: [`olmo_rl_compare/report.md`](https://huggingface.co/datasets/timf34/grader-shifts-persona-space-outputs/blob/main/olmo_rl_compare/report.md)
(figures in `olmo_rl_compare/figures/`; code `scripts/olmo_rl_compare.py`). Layer 36, paired persona bootstrap; Δ = final − DPO.

- **A little RL (step 100) changes nothing.** It matches DPO on every measure.
- **The persona landscape keeps its shape.** Prompt-token distance matrices correlate 0.975 [0.94, 0.99] between DPO and final; spread, dimensionality and the default Assistant's neighbours (evaluator, advocate, scientist) are unchanged. Answer-token landscapes keep their arrangement (0.985) but spread out ~60%.
- **Behaviour under grading changes.** On open-ended questions, "evaluated automatically" lowers role adherence only after RL: DPO +0.04, final −0.23 (difference −0.27 [−0.39, −0.15]). The length-matched neutral sentence has no effect; human grading moves the same way, about half as much.
- **Answers get much shorter under automated grading:** 0.51× baseline length after RL vs 0.86× before.
- **The answer-token grading shift grows ~3× and rotates, but it mostly tracks answer shortening** (r ≈ −0.75 across personas). Shorter answers leave less room for role-play, so part of the adherence drop may be length.
- **The prompt-token grading direction barely changes:** ~8% larger, same direction (cos 0.98), slightly more persona convergence.

**Reading:** RL does not create or reorganise the grader representation; it changes what the model does with it.

## In progress: steering across checkpoints

`configs/steering_olmo.yaml`: each checkpoint is steered along the **same** prompt-token grading directions, estimated on the DPO model (and, as a check, on the final model), at ±0.1/±0.3 × each model's mean residual norm, layer 36. Benchmarks: role adherence, School of Reward Hacks, agentic misalignment, against random directions. Question: does the direction only cause (bad) behaviour after RLVR?

## Reproduce

Per checkpoint, the pods run `scripts/pod/run_q1.sh <model key>` with these settings:
- `GSHIFT_EXPERIMENT=configs/experiment_olmo_rl.yaml`
- `STAGES=main`
- `SKIP_JUDGE=1`
- `SHARD=k SHARDS=2`

Judging and analysis then run on a laptop:
- `scripts/03_judge.py --model <key>`
- `scripts/04_analyze.py --model <key> --reps <rep> --out-dir …`
- `scripts/merge_analysis.py`
- `scripts/make_figures.py`

Every analysis script needs `GSHIFT_EXPERIMENT=configs/experiment_olmo_rl.yaml` set too. Tests: `tests/test_olmo.py`.
