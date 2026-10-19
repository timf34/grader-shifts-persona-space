# Handoff: finish design v3 + validation tooling

> **Status 2026-10-19 (second session):** items 1-4 and 6 below are done and merged
> (suite green, transformers pinned to 5.14.1, pod runners rebuilt on
> scripts/pod/_lib.sh incl. run_pilot.sh, pilot_report.py, design_report.py,
> gpu_checks.py + docs/gpu_validation.md, README). Design decisions taken with the user:
> examiner/reviewer/hacker/saboteur added by a seeded swap, the brief's pilot size with
> equal token budgets, the brief's w1 wording, wording-validation reuses main-grid
> baselines via `baseline_id` (counted once), HF datasets public by default. Item 5 (Q2
> additions) is done too: TruthfulQA + trait suites wired, trait judge, projection
> directions. Next: smoke test and pilot on the pod (README "Running on RunPod"). The
> README is the current reference; the text below is the original handoff.
>
> **2026-10-19 (later):** persona set expanded 50 -> 100 roles (20 per category; 80
> discovery / 20 held out), a user departure from the brief; original 50, held-out 10 and
> pilot roles unchanged (`role_expansion`, `role_split` in the config; README
> "Departures from the brief"). Main grid now 120,240 per model.

Branch: `claude/loving-clarke-4uxbqs` (WIP, **untested**). `main` = `b1aa2e8`, the last
end-to-end-working version (design v2). Merge the branch into `main` only once the test
suite passes. Do **not** launch the full grid.

## Goal (from the research brief "Exp 1. Split Brain + Assistant")

Q1: is there a shared grading-context direction across 50 Assistant-Axis personas, and
how does it move personas (H1 shared translation preserving separation / H2 convergence /
H3 persona-dependent shifts predicted by baseline position)? Q2: causal steering with
EasySteer. Models: Qwen3.6-27B and Qwen3.8-27B, thinking mode. Runs go on RunPod via the
user's `rp` runner (github.com/timf34/runpod-runner).

## What is in the branch

**Design v3 (`configs/experiment.yaml`, `src/gshift/design.py`, `src/gshift/config.py`)**
- 50 roles (40 discovery / 10 held out, stratified; `role_split.heldout`), 5 prompt
  variants, 60 questions (40 discovery / 20 validation, split by whole families, groups
  balanced; `data/questions/manifest_v1.jsonl`).
- Groups kept separate. `original` questions: `baseline, auto, human`. `checkable`
  questions: `baseline` + {automated, human} x {correctness, quality}
  (`auto_correct, human_correct, auto_quality, human_quality`). Each condition has
  `evaluator`/`criterion` fields.
- Wordings: 3 discovery templates (`w1..w3`) assigned deterministically per
  question-prompt-variant block (`(rank within group/split + pv) % 3`), identical across
  roles/models/matched conditions; baseline once per block (wording id `none`). 2 reserved
  validation wordings (`v1, v2`) used only in stage `wording_validation`
  (validation questions x pv0 x all roles).
- `request_id = role|p{pv}|qid|cond|wording|s{sample}`. Request carries evaluator,
  criterion, wording_id, wording_set, enable_thinking.
- Pilot stage: 10 discovery roles (seeded), pv [0,1], discovery questions only,
  thinking on (max_tokens 8192) and off (2048), outputs under
  `outputs_pilot/thinking_on|off` (`scripts/_common.py --thinking on|off`).
- Expected counts (verified by hand, put in tests): main 60,240 per model (120,480 for
  both, incl. default-assistant control); wording_validation 6,120 (3,060 per wording);
  followup 3,060; pilot 3,360 per thinking arm. Wording balance per group: 34/33/33
  (discovery), 17/17/16 (validation).

**Generation/extraction (`generate.py`, `extract.py`, `templates.py`)**
- `config_identity` hash now includes conditions, condition_sets, wording_templates,
  wording_assignment, followup_statement; a clash with existing outputs aborts (never
  silently reuses). `run_generation` refuses requests whose `enable_thinking` differs
  from the generation config.
- `TemplateInfo.thinking`; `split_generation` handles thinking-off output (no `</think>`
  -> whole visible output is the answer).
- `extract.META_FIELDS` adds evaluator, criterion, wording_id, wording_set,
  enable_thinking.

**Q1 analysis (all new, NOT yet run)**
- `analysis/contrast.py`: `ContrastData` (per-role x matched-unit contrast vectors,
  per-part weights), `from_cells`, `pooled`, `disp`, sign-flip `permutation_null`,
  roles x questions `bootstrap`.
- `analysis/shared.py`: alignment null, split-half stability, family leave-one-out,
  wording stability, cross-fitted dz.
- `analysis/geometry.py`: cross-fitted (independent whole-family halves) contraction
  coefficient beta, unbiased separation ratio, shared-energy fraction, LOO prediction of
  residual displacement from baseline position (linear on PCA ranks 3,5 vs isotropic),
  role-permutation null, role bootstrap, held-out-role / validation-question
  generalisation, verdict labels H1/H2/H3. Skips non-two-state contrasts.
- `analysis/report.py` rewritten around these; persona PCA on discovery-role baselines
  only, PC1 vs Assistant-contrast alignment with bootstrap CI (only labelled "Assistant
  Axis" above `analysis.assistant_alignment_threshold`); pooled contrasts explicitly
  labelled; validation-wording and follow-up blocks.
- `tests/synthetic.py` / `tests/test_analysis.py` rewritten for v3 (planted u_g, u_e,
  u_c, contraction alpha).

**Q2 extras (partial)**
- `src/gshift/steer/extra_benchmarks.py` + `data/truthfulqa/` (MC1, 817 rows) +
  `data/trait_openended/` (200 questions, frozen rubric, sha256-checked). Held-out only,
  never used for calibration. **Not wired in yet**, and it references
  `gshift.steer.trait_judge`, which does not exist yet.

## Remaining work, in priority order

1. **Get the suite green.** `pip install -e ".[dev,judge]"`, then
   `python -m pytest tests -q -p no:warnings` (tokenizer-dependent tests need
   `python scripts/fetch_tokenizers.py` once, or set `HF_HOME`). Expect failures in:
   - `tests/test_analysis.py` (new analysis code never run) - fix code, not thresholds,
     unless a threshold is clearly mis-set for the synthetic world.
   - `tests/test_design.py` - still asserts v2 counts / stage `paraphrase`. Update to the
     counts above, per-group condition sets, wording balance, seeds shared between
     `main` and `wording_validation`, prefix format `"<statement>\n\n<question>"` with
     the block's wording, default-assistant control has no system message.
   - `src/gshift/analysis/crossmodel.py` - still uses the old `Cells.matched` structure.
     Rebuild on `ContrastData` via `report._group_cells` + `contrast.from_cells`. Test
     expects `res["primary"][rep]["<group>|<contrast>"]` with
     `"paired_bootstrap_b_minus_a"` (`g_norm_over_act_norm` with `lo`) and
     `"second_order"` containing `"rsa_persona_distances_baseline"`. Role centroids are
     no longer saved by report.py: either save `{group}_{rep}_L{L}_role_centroids.npy`
     there or compute RSA from cells directly.
2. **Pin the stack.** `requirements/pod.txt` has `transformers>=5.5.3,<5.15` (untested
   range). Pick and pin an exact version that has `qwen3_5`
   (`Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`; "Qwen3.5" is the
   architecture name of both Qwen3.6 and Qwen3.8, not a wrong target), confirm
   `AutoConfig` + tokenizer + chat template at both pinned revisions in
   `configs/models.yaml`, and add a test + a note in the README. The user previously
   rejected a downgrade to 5.16.1 - ask before changing major/minor direction.
3. **Pod scripts.** `scripts/pod/run_q1.sh`: default `STAGES` is still
   `main paraphrase followup` -> `main wording_validation followup`. Add
   `scripts/pod/run_pilot.sh` (both thinking arms, discovery only, separate output
   roots) and `scripts/pilot_report.py` (truncation, token lengths, adherence,
   displacement reliability via split halves / prompt-variant halves, timings).
   Add `scripts/design_report.py` (counts per stage/model, wording balance, config ids).
4. **GPU validation tooling** (user runs it): scripts + `docs/gpu_validation.md` for
   (a) zero-strength EasySteer output == unsteered output; (b) a nonzero intervention
   reaches the intended layer and token positions with correct sign, scale
   (`strength * A_L * unit(v)`) and GGUF conversion - compare HF hidden states with an
   added vector against EasySteer, remembering **our layer L = EasySteer layer L-1**;
   (c) token-span parsing (`<think>` 248068, `</think>` 248069, `<|im_end|>` 248046) on
   real generations; (d) exact-token replay check. Keep these CPU-testable where possible
   (`tests/test_gpu_code_on_cpu.py` pattern).
5. **Q2 additions.**
   - Wire `extra_benchmarks` into `steer/tasks.py` (`benchmark_prompts`),
     `steer/report.py` (primary metrics: TruthfulQA accuracy, trait scores),
     `scripts/12_steer_judge.py` (trait judge; write `steer/trait_judge.py`), and a
     `benchmarks` block in `configs/steering.yaml`. Assert calibration prompts never
     include TruthfulQA/trait items.
   - Projection directions in `configs/steering.yaml` + `steer/vectors.py`: project the
     grading direction g onto the baseline persona PCA subspace (saved by report.py as
     `vectors/{group}_{rep}_L{L}_baseline_pcs.npy`, `_baseline_pca_center.npy`) at
     predefined ranks (e.g. 5, 20) and its orthogonal residual; steer each at the
     original contribution (raw `P g`, not renormalised) and at matched norm (`||g||`).
     Skip the matched-norm version when `||P g|| / ||g|| < 0.1` (record why). Label
     results "inside/outside the baseline PCA subspace", never "outside all persona
     representations".
6. **README.md**: setup with `rp` (two venvs: HF extraction venv and EasySteer venv,
   `vllm==0.26.0` + EasySteer overlay `ZJU-REAL/EasySteer-vllm-v1@6267ca0`, cu128 if the
   driver is 12.x), pinned stack, smoke / pilot / full commands, resume behaviour and
   config ids, output schemas (responses jsonl, activation dirs, summary.json,
   vectors), layer convention, known limitations, a table of checks passed on CPU vs
   checks that need a GPU, departures from the brief (below).
7. Commit, merge into `main`, push. Final report to the user: changes, exact checks
   passed, remaining blockers, run counts + config hashes per model, smoke and pilot
   commands, departures from the brief.

## Conventions you must keep

- **Layer L** = residual stream at the INPUT of decoder block L = HF `hidden_states[L]`
  (Betley convention). EasySteer layer k = output of block k, so our L = EasySteer L-1;
  GGUF tensor name `direction.{L-1}`. Primary layer 36; sweep [16,24,32,36,40,48,56].
- **Steering scale**: `strength * A_L * unit(v)`, `A_L` = mean residual norm at L
  (from the Betley repro report).
- **Models** (`configs/models.yaml`): Qwen3.6-27B @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
  Qwen3.8-27B @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. 64 blocks, hidden 5120.
  Qwen3.8 uses `chat_template_kwargs: {reasoning_effort: medium}` (otherwise its template
  injects "Reasoning effort is set to xhigh"). Matched prompt structure does not imply
  matched reasoning compute - record rendered templates and settings.
- **Exact-token replay**: vLLM generates from rendered token ids; HF re-runs
  `prompt_ids + gen_ids` teacher-forced, batch size 1 (GatedDeltaNet layers, no padding),
  through the text decoder only. Reps: `pre` (last prompt token), `resp` (final-answer
  mean), `reason` (reasoning mean).
- **EasySteer client** (user-verified): `vllm serve ... --enable-steer-vector
  --steer-algorithms direct --steer-graph-mode split`; OpenAI completions with token-id
  prompts, `extra_body={"steering": spec, "return_token_ids": True,
  "skip_special_tokens": False, "top_k": .., "min_p": ..}`; omit `steering` for
  unsteered requests.
- Already verified on CPU in `b1aa2e8`: hooks == `hidden_states`; a VL checkpoint reloads
  as `Qwen3_5ForCausalLM` with no missing keys (tiny model); Qwen3.6 rendered prompt ==
  Betley's byte for byte; Qwen3.8 `medium` injects nothing; tiny-model extraction ==
  HF hidden states with the real tokenizer; Betley comparison detects an off-by-one layer.

## Departures from the brief so far (put in README)

- Q2 "natural" grading prefixes use discovery wording `w1`.
- Wording-validation panel = validation questions x prompt variant 0 x all 50 roles
  (frozen, 6,120 runs).
- Pilot outputs are kept in a separate root and never feed confirmatory analysis.
- Qwen3.8 `reasoning_effort: medium` is provisional.

## Rules

Never commit `.env`, API keys or deploy keys. No model identifiers in commit messages.
Don't launch the full grid. Don't skip/disable tests to get green.
