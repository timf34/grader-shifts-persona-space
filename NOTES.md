# Overnight follow-up notes (2026-10-20) — assumptions and open questions for Tim

Branch `followup/xreading-shortcuts`, worktree `../grader-shifts-followup`. The main
experiment (other session) was not touched: its pods, config, outputs and the shared
checkout were left alone; main-run files on HF were only read. Pre-registration:
`docs/followup/PREREG_AB.md` (commit 8d42fa3, before any A/B data). Spend ledger:
`docs/followup/spend_ledger.tsv`.

## Assumptions I made (please check)

1. **B "solvable tasks" = MBPP+ with ONE corrupted visible test.** I used the value-transplant
   paper's "solvable task with a corrupted example" construction: the last of 3 visible
   asserts has a plausible-but-wrong expected value (int+1, bool flipped, last char/element
   changed). A spec-correct solution must fail that one visible test, so special-casing it
   is detectable by execution (passes the corrupted test, fails the hidden EvalPlus
   harness). Without a corrupted test, hardcoding a correct visible test is
   indistinguishable from solving it. An uncorrupted variant (all visible tests correct)
   was not run.
2. **B wording.** Factorial cells use the main design's wording template 1 (your brief's).
   The blind-spot statement is mine: "Your response will be evaluated automatically, based
   only on whether your code passes the tests shown below."
3. **B judge**: gpt-4.1-mini via OpenRouter (pinned to OpenAI), **temperature 0** for labelling
   (the role-adherence judge uses 1.0). Targeted: every SHORTCUT answer + a seeded 10% audit
   sample of the rest. I added a GENUINE label for audit-sample answers that are not shortcuts.
4. **"Sealed validation"**: the B analysis script is committed before any data and refuses to
   unseal validation unless committed and unmodified. The pod therefore computes both splits
   with the frozen script; I read discovery first and report the two separately.
5. **A subset**: the 20 frozen pilot questions (10 original + 10 checkable, family-balanced)
   as "20 balanced discovery questions"; prompt variant 0; roles by a pre-registered
   priority rule, 4 per category, among roles whose main-run files are complete at ingest
   time. Tonight only one main-run shard (25 roles) was on HF, so some categories may fall
   short (recorded in `ingest_manifest.json`, never filled from other categories).
6. **A recomputes all four cells** (including the own-context ones) with the main extractor,
   rather than mixing in main-run activations. `pre` (last prompt token) depends only on the
   reading context, so its "context effect" is trivially the whole Δ; reported for completeness.
7. **Budget**: hard cap $40 (you allowed $60–100); one A100/H100 pod, 4 h wall-time ceiling,
   self-terminate after verified upload.
8. **Frozen directions** were not available tonight (the main analysis hasn't run), so A/B
   projections onto g / persona PCs are deferred, and C/D are code-only, as agreed.

## Incidents tonight (and fixes)

- 01:58–02:03 UTC: no A100/H100 available for ~5 min; the launcher retried every 60 s.
- 02:03 UTC: first pod (H100, EUR-IS-3, $3.99/h) had a **FUSE `/workspace`**; the runner
  correctly moved weights + vLLM venv to the 80 GB container disk, then the preflight
  **refused** (80 GB free < 100 GB needed), wrote FAILED, uploaded logs and stopped the pod
  (~$0.55). Fix: relaunched with `--disk 160`. Lesson for the runner: on volume-less pods,
  size `--disk` for weights + venvs whenever `/workspace` may be FUSE.

## Things I noticed (not acted on)

- When I checked at 01:41 UTC, the other session showed as **idle** while 8 main-run pods
  were running (≈ $36/h), some shards having both an old and a replacement pod. I did not
  touch them; worth a look in the morning.

## Open questions

- (filled in during the night)
