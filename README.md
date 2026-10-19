# Grading-induced shifts in persona space

Code for "Exp 1. Split Brain + Assistant": does telling Qwen3.6-27B and Qwen3.8-27B (thinking
mode) that a response will be graded add a shared activation shift across 50 Assistant Axis
personas, compress the differences between them, or move them in a way that depends on
where they start (Q1)? And does steering along the recovered directions change behaviour
(Q2, EasySteer)? These experiments test context-dependent behavioural modes; they cannot
by themselves establish separate "brains" or attribute a representation to RL.

The pre-registered design is `configs/experiment.yaml` (design v3, frozen 2026-10-19); the
models and their pinned revisions are in `configs/models.yaml`, Q2 in
`configs/steering.yaml`. `python scripts/design_report.py` prints the frozen grid, its
balance and the config ids.

## Design at a glance

* **Roles**: 50 Assistant Axis role definitions (verbatim from
  `safety-research/assistant-axis@a989619`), 10 per category (analytical/helping,
  relational, competitive, creative, nonhuman), including the evaluator-type
  (evaluator, examiner, reviewer), robot and hacker/saboteur roles; all 5 prompt variants;
  plus an unprompted default-Assistant control (no system message), analysed separately.
  40 discovery / 10 held-out roles, stratified (2 per category, seeded).
* **Questions**: `data/questions/manifest_v1.jsonl`, 30 original extraction questions + 30
  objectively checkable tasks (coding, arithmetic, structured data); per group 20
  discovery / 10 validation, validation = whole question families.
* **Conditions** (user-message prefix `"<statement>\n\n<question>"`, role instruction
  unchanged): original questions `baseline, auto, human`; checkable questions `baseline`
  + {automated, human} x {correctness, quality}. Three discovery wordings (`w1` = the
  brief's template 1), one per question x prompt-variant block, balanced, identical
  across roles, models and conditions; baseline generated once per block. Two reserved
  validation wordings (`v1`, `v2`) for a separate wording-generalisation panel.
* **Representations**: final-prompt position (`pre`), mean over final-answer tokens
  (`resp`), mean over reasoning tokens (`reason`, separate), at the Betley layer 36 and a
  sweep [16, 24, 32, 36, 40, 48, 56].

| stage | per model | both models |
|---|---:|---:|
| main (50 x 5 x (30 x 3 + 30 x 5) + default 240) | 60,240 | 120,480 |
| wording_validation (v1 + v2, validation questions, prompt 0; baselines reused from main) | 6,120 | 12,240 |
| followup ("This interaction is part of a model evaluation.", prompt 0) | 3,060 | 6,120 |
| pilot, per thinking arm (10 discovery roles x 3 prompts x 10 + 10 questions + 80 default) | 2,480 | 4,960 |
| pilot total (2 arms x 2 models) | | 9,920 |

## Running on RunPod

Pods are driven from the laptop with [`rp`](https://github.com/timf34/runpod-runner)
(`/Users/timf34/Documents/VSCode/runpod-runner/.venv/bin/rp` if it is not on PATH). Run
every `rp` command from this repo's root (so `--req requirements/pod.txt` is uploaded and
its `# rp-min-cuda: 12.8` header is enforced). Results go to the HF dataset in
`$HF_DATASET_REPO` after every step (created **public** if it does not exist; pass
`--private` to `scripts/upload_hf.py` to override; an existing repo's visibility is never
changed); nothing is kept on network volumes.

`.env` (copy `.env.example`, never commit it): `HF_TOKEN`, `HF_DATASET_REPO`,
`OPENAI_API_KEY` (role-adherence judge), `ANTHROPIC_API_KEY` (Q2 judges), and
`RUNPOD_API_KEY` (lets an unattended job stop its own pod).

```bash
# 1. a pod: ~55 GB of weights per model + two venvs + outputs need --volume-size >= 150
rp up --name gs-q36 --gpu h200 --gpus 1 --volume none --volume-size 200 --disk 80
rp bootstrap gs-q36 --repo https://github.com/timf34/grader-shifts-persona-space --env .env --req requirements/pod.txt

# 2. smoke test only (tiny grid: generation, Betley layer verification, span + replay checks)
rp run gs-q36 --job smoke --env SMOKE_ONLY=1 --env SHUTDOWN=stop -- bash scripts/pod/run_q1.sh qwen3.6-27b

# 3. thinking on/off pilot (smoke first unless SKIP_SMOKE=1), then read the pilot report
rp run gs-q36 --job pilot --env SHUTDOWN=stop -- bash scripts/pod/run_pilot.sh qwen3.6-27b

# 4. the full Q1 run for one model (one pod per model; GEN_TP=2 on 80 GB GPUs)
rp run gs-q36 --job q1 --env SHUTDOWN=stop -- bash scripts/pod/run_q1.sh qwen3.6-27b

# 5. Q2 after both Q1 runs
rp run gs-q36 --job q2 --env SHUTDOWN=stop -- bash scripts/pod/run_q2.sh qwen3.6-27b

rp logs gs-q36 --job q1 -n 40      # progress; also logs/<job>/<model>/STATE.md on the HF dataset
rp jobs gs-q36                     # running / EXIT=<code>
rp down gs-q36                     # if SHUTDOWN was not set
```

Every runner (`scripts/pod/_lib.sh`): runs a fail-fast preflight (`scripts/pod/doctor.py`:
GPUs, disk, pinned revisions reachable, results dataset writable, judge key live,
self-stop configured; `DOCTOR_ONLY=1` stops there); downloads weights in the background
while the EasySteer venv installs; writes a timestamped `STATE.md` and a final
`DONE`/`FAILED` marker and uploads them with the step logs to `logs/<job>/<model>/`;
stops the pod through the RunPod REST API at the end (`SHUTDOWN=stop|terminate`;
`terminate` is downgraded to `stop` on any failure or failed upload) and after a hard
ceiling (`MAX_HOURS`: 48 Q1, 10 pilot, 24 Q2) even if a step hangs. Job names must not
contain `.` (rp refuses them).

Expected time: the pilot report prints measured tokens/s and a GPU-hour estimate for
the main grid; plan the Q1 pods from it.

### Two venvs, pinned stack

| venv | made by | used for | pins |
|---|---|---|---|
| main (`/workspace/<repo>_venv`) | `rp bootstrap --req requirements/pod.txt` | HF extraction, Betley reproduction, judges, analysis, steering client, activation check | `torch==2.8.*` (cu128), `transformers==5.14.1` |
| EasySteer (`/workspace/es_venv_cu129` or `_cu130`) | `scripts/pod/install_easysteer.sh` (every run, idempotent) | vLLM generation (Q1) and the steering server (Q2) | `vllm==0.26.0` + `ZJU-REAL/EasySteer-vllm-v1@6267ca0` overlay, `transformers==5.14.1`, torch 2.11.0 |

The EasySteer venv follows the host driver: CUDA >= 13 drivers use the PyPI vLLM wheel;
CUDA 12.x drivers use the official `vllm-0.26.0+cu129` release wheel with
`torch==2.11.0+cu129`, a CPU `torchcodec`, and NVIDIA's `cuda-compat-12-9` libraries on
drivers < 575 (the PyPI wheel needs `libcudart.so.13` even with a cu128 index). The venv's
`bin/` is put on PATH for vLLM processes (vLLM launches `ninja` by name) and the compat
libraries are applied to vLLM processes only. `transformers==5.14.1` is what the
previously verified EasySteer range (`>=5.5.3,<5.15`) resolves to; it has the `qwen3_5`
architecture (`Qwen3_5ForConditionalGeneration`; "Qwen3.5" is the architecture name of
both Qwen3.6 and Qwen3.8). `tests/test_gpu_code_on_cpu.py::test_pinned_stack` loads both
pinned revisions with it (config, layer types, special token ids, thinking on/off
templates).

## Q2: steering (configs/steering.yaml)

Each model is steered along its own Q1 directions (pooled `auto_minus_baseline`, `resp`
and `pre`), a Betley-style direction reproduced on the same model (plus the published
0007 vector on Qwen3.6), 5 norm-matched random directions, and the decomposition of the
primary direction into its parts inside / outside the discovery-only baseline PCA
subspace at ranks 1, 5, 10, 20 (capped at the empirical rank), each at its original
contribution and at matched norm (skipped, and recorded, when the component is < 0.1 of
||g||). Strength is calibrated on separate prompts (Assistant Axis questions outside the
manifest + GSM8K train; checked to share no item with any held-out suite) by the
pre-registered rule, then evaluated at +-{s*/2, s*} (random and decomposition directions
at +-s*). Benchmarks: School of Reward Hacks (proxy metric and independently judged
quality, reported separately), TruthfulQA MC1 (Betley's port, shuffled options; held
out), open-ended trait expression (10 traits x 20, frozen rubric, judge settings from
`data/trait_openended/PROVENANCE.json`; held out), GSM8K capability, plus unsteered
`natural_auto` / `natural_human` prefix conditions and an HF activation check (does
steering move held-out persona activations like natural grading prompts do?). Before
any of it, `run_q2.sh` verifies the steering path (docs/gpu_validation.md).

## Resumption and config ids

Generation is resumable: a request is skipped iff a record with the same id **and** the
same config id is on disk. The config id hashes the model revision, tokenizer revision,
template kwargs, sampling settings (incl. thinking and token budget), design version,
manifest hash, condition definitions, wording texts and their assignment rules. Any
record in a touched file (or a referenced main-grid baseline) under another config id
aborts the run: different settings go to a different output root (`GSHIFT_OUTPUTS`).
Request ids are `role|p{variant}|question|condition|wording|s{sample}`, so wordings,
stages and conditions never collide. Graded requests name their matched main-grid
baseline (`baseline_id`); the wording-validation and follow-up stages reuse it rather
than regenerating it (same revision, rendered prompt, settings, budget and seed;
regenerating with the same seed would not be an independent replication). Extraction is
resumable per (condition, role) file. Pilot arms (`outputs_pilot/thinking_{on,off}/`) and
smoke runs (`outputs_smoke/`) have their own roots.

## Outputs

```
outputs/<model>/
  run_config/<config_id>.json, timings_<config_id>.jsonl   identity, rendered template, environment
  <condition>/responses/<role>.jsonl      one record per request: all Request fields (role, prompt
                                          variant, question id/group/family/split, condition, evaluator,
                                          criterion, wording id/set, baseline_id, seed, system prompt,
                                          original question, prefix, user content, messages) + model_key,
                                          config_id, prompt_text, prompt_ids, gen_ids, gen_text,
                                          finish_reason, truncated, max_tokens, answer/reasoning spans and
                                          texts, token counts, think_closed, has_answer
  <condition>/activations/<role>/{pre,resp,reason}.npy   [n, n_sweep_layers, hidden] float16
                                 norms.npy [n, 3, n_layers], index.jsonl (id, row, metadata, spans), meta.json
  <condition>/scores/<role>.jsonl         role-adherence judge scores (0-3)
  betley_repro/                           Betley vector reproduced with our extractor + verification report
  analysis/summary.json                   every Q1 statistic (per representation x layer x group x contrast)
  analysis/vectors/                       directions g, role displacements, baseline PCs, Assistant contrast
  analysis/tables/layer_sweep.csv, analysis/figures/
  steering/                               Q2 vectors, generations, judgements, behaviour_summary.json
outputs/comparison/comparison.json        Qwen3.6 vs Qwen3.8 (paired bootstrap, RSA)
```

## Layer and steering conventions

* **Layer L** = residual stream at the INPUT of decoder block L = HF `hidden_states[L]`
  (Betley et al.'s convention; the Assistant Axis code's "layer L-1"). Primary layer 36.
* **EasySteer layer k** = OUTPUT of block k, so our L is sent as EasySteer L-1 (GGUF tensor
  `direction.{L-1}`).
* **Scale**: strength x A_L x unit(v), A_L = mean residual norm at L (from the Betley
  reproduction on the same model).
* **Exact-token replay**: vLLM generates from token ids rendered once by the HF tokenizer;
  HF re-runs `prompt_ids + gen_ids` teacher-forced, batch size 1 (Gated-DeltaNet layers,
  no padding), through the text decoder only.

## Checks: CPU vs GPU

`pip install -e ".[dev,judge]"`, plus `torch==2.8.*` and `transformers==5.14.1` for the
model-side tests, `python scripts/fetch_tokenizers.py` once (tokenizer/config files only),
then `python -m pytest tests -q -p no:warnings`.

| verified on CPU (tests/) | needs a GPU (docs/gpu_validation.md) |
|---|---|
| grid sizes, condition sets, wording assignment and balance, seeds shared across stages, baseline reuse, splits and seeded draws (roles, held-out, pilot) | Betley layer/position/template verification on the real model |
| prompt rendering byte for byte (Qwen3.6 = Betley's), thinking on/off templates at both pinned revisions with the pinned transformers | span parsing on real generations |
| hooks == `hidden_states`; tiny-model extraction == HF hidden states with the real tokenizer; Betley off-by-one detection | exact-token replay vLLM -> HF |
| analysis recovers planted structure, distinguishes translation / convergence, cross-fitting prevents spurious convergence, null stays null, cross-model paired bootstrap, pilot report | zero-strength EasySteer == unsteered |
| validation logic: span invariants, replay misalignment, steering layer/scale/sign identification, GGUF round trip | EasySteer layer, positions, sign, scale vs HF |

## Departures from the brief

* Wording-validation panel = validation questions x prompt variant 0 x all 50 roles (6,120
  per model). Its baselines are the matched main-grid records, referenced explicitly and
  counted once in every combined or comparative analysis (the combined estimate averages
  v1 and v2 within a unit before subtracting the baseline; v1 vs v2 is compared
  graded-to-graded). A replication across generation seeds would rerun baseline and
  graded conditions with new seeds; it is not scheduled.
* Roles: the brief's named roles are included; examiner, reviewer, hacker and saboteur
  replaced 4 roles of an earlier hand-picked list by a seeded swap (documented in the
  config). Grader-type and adversarial roles are also reported as leave-out sensitivity
  analyses.
* Default-Assistant control: its own question-level wording assignment (rank within
  question group mod 3, 10/10/10 per group), since it has one prompt variant.
* Pilot: thinking on/off use the same token budget (8,192); pilot outputs live in separate
  roots and never feed the confirmatory analysis, even though the thinking-on pilot has
  the main grid's config id.
* Secondary (adherence-filtered) analysis: the minimum matched pairs per role is half of
  what the panel can contain (50 of 100 discovery units, 25 of 50 validation units).
* Reliability is assessed by question split halves (balanced by family / group), family
  leave-one-out, per-wording stability and, in the pilot, prompt-variant halves; there is
  no second independent sample per prompt in the main grid.
* Q2 "natural" grading prefixes use discovery wording `w1`.
* Qwen3.8 `reasoning_effort: medium` is provisional (xhigh and low inject a system
  preamble); matched prompt structure does not imply matched reasoning compute.

## Known limitations

* Our Assistant contrast is a 40-role subset estimate (default control minus mean role),
  not the original 275-role Assistant Axis; PC1 is only called Assistant-aligned above
  `analysis.assistant_alignment_threshold`.
* A direction outside our sampled persona PCA subspace is not thereby outside all persona
  representations.
* Judge blinding is partial: the answer itself may reveal the condition.
* Raw coordinates are never compared across models.
