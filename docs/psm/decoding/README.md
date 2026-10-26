# Persona-identity decoding under grading (Q1 data, layer 36)

Question: does the grading prefix hide persona information or weaken it? 100-way decoding of the 100 prompted roles (`default` excluded), open-ended questions only, from `outputs/<model>/{baseline,auto,human}` (auto/human = wordings w1-w3, each paired 1:1 with a baseline record). `full` = ridge classifier on standardised 5120-d features (λ by grouped inner CV on train; probabilities by a temperature fitted on out-of-fold train scores). `pc20`/`pc5` = multinomial logistic on the top-k PCs of the baseline-TRAIN role centroids (α by inner CV). Chance = 0.01.
Tests: **i** base→base, **ii** base-trained→auto, **iii** auto→auto, **ii-c** = ii after subtracting the mean auto−baseline shift (from train data). **A** = train discovery questions (20), test validation questions (10); **B** = train prompt paraphrases {0,1,2}, test {3,4}. Top-1 accuracy shown; mean P(correct) and 95% bootstrap CIs (over test questions, paired across tests) are in `results.json`. Rerun: `MallocLargeCache=0 ~/.claude/bin/memguard -m 8 -- python decoding.py --model M --layers 36` (peak 2.6 GB/slice), then `--merge`.
Data checks: index `row` = position in every file; `pre` norms match `norms.npy` to 2e-7 (row alignment holds). 2-3 records per model with no answer are dropped from `resp`.

**qwen3.6-27b**
| rep / space | A: i | ii | iii | ii-c | iii−i [95% CI] | B: i | ii | iii | ii-c | human A: ii / iii |
|---|---|---|---|---|---|---|---|---|---|---|
| resp full | 0.86 | 0.70 | 0.74 | 0.70 | -0.12 [-0.16, -0.07] | 0.94 | 0.85 | 0.91 | 0.87 | 0.84 / 0.85 |
| resp pc20 | 0.58 | 0.40 | 0.44 | 0.41 | -0.14 [-0.17, -0.12] | 0.72 | 0.53 | 0.61 | 0.56 | 0.52 / 0.52 |
| reason full | 0.94 | 0.83 | 0.90 | 0.87 | -0.05 [-0.08, -0.02] | 0.97 | 0.82 | 0.97 | 0.92 | 0.95 / 0.96 |
| reason pc20 | 0.72 | 0.55 | 0.59 | 0.56 | -0.13 [-0.16, -0.10] | 0.82 | 0.64 | 0.72 | 0.69 | 0.69 / 0.68 |
| pre full | 1.00 | 0.93 | 1.00 | 0.99 | +0.00 [+0.00, +0.00] | 1.00 | 0.83 | 1.00 | 0.97 | 0.99 / 1.00 |
| pre pc20 | 0.95 | 0.40 | 0.90 | 0.76 | -0.05 [-0.07, -0.03] | 0.91 | 0.39 | 0.85 | 0.72 | 0.69 / 0.94 |

**qwen3.8-27b**
| rep / space | A: i | ii | iii | ii-c | iii−i [95% CI] | B: i | ii | iii | ii-c | human A: ii / iii |
|---|---|---|---|---|---|---|---|---|---|---|
| resp full | 0.76 | 0.71 | 0.72 | 0.71 | -0.03 [-0.06, -0.01] | 0.89 | 0.82 | 0.86 | 0.83 | 0.76 / 0.75 |
| resp pc20 | 0.44 | 0.44 | 0.43 | 0.43 | -0.00 [-0.02, +0.01] | 0.57 | 0.54 | 0.56 | 0.55 | 0.44 / 0.44 |
| reason full | 0.94 | 0.84 | 0.89 | 0.86 | -0.05 [-0.08, -0.01] | 0.97 | 0.82 | 0.97 | 0.91 | 0.91 / 0.93 |
| reason pc20 | 0.69 | 0.60 | 0.62 | 0.62 | -0.07 [-0.11, -0.02] | 0.72 | 0.64 | 0.67 | 0.67 | 0.67 / 0.67 |
| pre full | 1.00 | 0.86 | 1.00 | 1.00 | +0.00 [+0.00, +0.00] | 0.99 | 0.61 | 0.99 | 0.97 | 0.99 / 1.00 |
| pre pc20 | 0.97 | 0.24 | 0.95 | 0.75 | -0.02 [-0.04, +0.00] | 0.90 | 0.32 | 0.86 | 0.71 | 0.46 / 0.96 |

## What the numbers establish
- **In generated text states, "auto" grading weakens persona identity rather than hiding it, clearly so in qwen3.6.** A decoder trained on auto data (iii) recovers less than the baseline decoder on baseline (resp −0.12 full, −0.14 pc20, new questions). Removing the shared shift recovers almost nothing (ii-c ≈ ii). The auto-trained decoder still reads baseline data at baseline accuracy (0.86), so the graded code is a weaker version of the same code, not a new one. In the PC20 subspace, between-role variance falls to 0.73× and within-role variance rises to 1.40× (qwen3.6 resp).
- **qwen3.8 shows much less:** resp is roughly unchanged (−0.03 full, 0.00 pc20). Reasoning loses a little (−0.05 full, −0.07 pc20).
- **At the prompt token, identity is moved, not lost.** A graded-trained decoder stays at ceiling (iii = 1.00 full, 0.90-0.95 pc20). The baseline decoder fails in low dimensions (pc20 ii 0.40 / 0.24). Subtracting the shift only partly rescues it (0.76 / 0.75), so the change is more than one translation. The 0.55-0.75× contraction at `pre` leaves decodability intact.
- **"Human" grading costs almost nothing in resp/reason** (ii and iii within about 0.02 of i). At `pre` it re-encodes identity like auto, but less strongly.
- **Results carry over across layers and wordings.** Layers 24/48 (scheme A) show the same pattern: qwen3.6 resp iii−i = −0.13 / −0.11; qwen3.8 resp −0.02 / −0.04. For resp, the drop survives answer-length matching (ratio ≤ 1.3, about half the pairs): qwen3.6 i 0.89 → ii 0.74, iii 0.78.

## Uncertain
- Scheme A tests only 10 questions, so absolute accuracies have wide CIs (e.g. qwen3.6 resp i 0.80-0.91). The paired deltas are tighter, but they come from the same 10 questions.
- Weaker identity in `resp` means could reflect more uniform, genre-like answer text under grading rather than a change in internal persona state; token means cannot separate the two. `reason` under scheme B is unaffected in full dimension (iii = i = 0.97). The loss shows up only for new questions and in the leading persona axes.
- The PC-k decoders use baseline PCs, so pc-k (iii) could understate persona axes specific to auto; the full-dimensional (iii) guards against this. Held-out wordings v1/v2 give larger drops (qwen3.6 resp ii 0.61, iii 0.65), so the effect size depends on the wording.
- One sample per configuration. Full-dimensional accuracy is not saturated for resp/reason, but pc5 accuracies are low (0.12-0.44; chance 0.01) with smaller deltas, so pc20 is the informative low-dimensional view (pc5 numbers are in results.json).
