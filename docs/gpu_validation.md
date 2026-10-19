# GPU validation

What has to be true on real hardware before any result is trusted, how each check is
run, and what it guards against. The check logic lives in `src/gshift/validation.py` and
is unit-tested on CPU with a tiny Qwen3.5-architecture model
(`tests/test_gpu_code_on_cpu.py`): every check passes on correct inputs and fails on a
planted mistake (stale stored spans, a replay misaligned by one token, steering applied
one layer early). On the pod, `scripts/gpu_checks.py` runs the same logic on the real
model and writes `<outputs root>/<model>/gpu_checks/<check>.json`; a failed check exits
non-zero and the pod runner stops before spending GPU time on the grid.

| check | what it proves | runs in | how |
|---|---|---|---|
| Betley verification | our layer index, extraction position and chat template match Betley et al.'s published Qwen3.6-27B vector (an off-by-one layer is detected) | every smoke run, every pilot / Q1 run (Qwen3.6: `--require-verification` stops the run) | `02_extract.py --betley-only --require-verification` |
| (c) span parsing | reasoning / final-answer spans are located correctly on the model's real output format (`<think>` 248068, `</think>` 248069, `<|im_end|>` 248046), thinking on and off: stored spans equal a fresh re-split, "stop" ends with an end token, truncation happens at the budget, no special tokens inside spans, no reasoning span with thinking off | smoke (run_q1 / run_pilot) | `gpu_checks.py spans --smoke` |
| (d) exact-token replay | the HF teacher-forced pass sees exactly the tokens vLLM generated from (prompt_ids + gen_ids, no re-templating): with top-k=20 sampling every sampled token must be in HF's top-20 at its position (>= 99%), and a shifted-by-one control must be clearly worse | smoke | `gpu_checks.py replay --smoke` (16 records across lengths, first 2,048 generated tokens) |
| (a) zero-strength steering | an EasySteer request with a scale-0 `direct` spec returns exactly the unsteered tokens, greedy and with a fixed sampling seed | run_q2 before calibration | `gpu_checks.py steer-server` (server up) |
| (b) intervention placement | with strength x A_L x unit(v) at our layer L (= EasySteer layer L-1, GGUF tensor `direction.{L-1}`), HF teacher-forced on the server's greedy tokens reproduces the server's per-token log-probs better than layer L-1, L+1, scale x0.5, x2 or the opposite sign, in every steered case; the GGUF payload round-trips to unit(v) | run_q2 before calibration | `gpu_checks.py steer-server`, stop the server, `gpu_checks.py steer-hf` |

The steering checks use A_L and the layer-L Betley delta from the Betley reproduction
under the same output root (random direction if the delta is missing), at strength 0.5
(`--strength`), on three short prompts with thinking on.

## Running them by hand

```bash
# smoke grid (tiny design, outputs_smoke/), then spans + replay
es_py scripts/01_generate.py --model qwen3.6-27b --smoke --stages main     # EasySteer venv python
python scripts/02_extract.py --model qwen3.6-27b --smoke --require-verification
python scripts/gpu_checks.py spans  --model qwen3.6-27b --smoke
python scripts/gpu_checks.py replay --model qwen3.6-27b --smoke

# steering path (needs Q1 outputs/<model>/betley_repro, or --smoke after the smoke run)
bash scripts/pod/serve_easysteer.sh qwen3.6-27b
python scripts/gpu_checks.py steer-server --model qwen3.6-27b --smoke
bash scripts/pod/serve_easysteer.sh stop
python scripts/gpu_checks.py steer-hf --model qwen3.6-27b --smoke
```

`run_q1.sh` and `run_pilot.sh` run the smoke grid, Betley verification, spans and replay
automatically; `run_q2.sh` runs (a) and (b) before any calibration or evaluation.

## Reading failures

* **spans**: `stored_span_differs_from_resplit` means records were written by different
  parsing code (regenerate or re-extract); `stop_without_end_token` or
  `truncated_but_not_at_budget` means the vLLM output handling changed. Warnings such as
  `multiple_think_close` are model behaviour and are reported, not fatal.
* **replay**: a low `frac_in_top_k` with a similar control means the replayed tokens are
  misaligned (re-templating, a dropped BOS or special token, or a different revision);
  check `prompt_sha` and the run config.
* **steer-hf**: `best == "L-1"` or `"L+1"` is a layer-convention error (EasySteer layer k
  is the OUTPUT of block k, so our L must be sent as L-1); `L_x0.5` / `L_x2` is a scale
  error (A_L or a double normalisation); `L_flipped_sign` is a sign error; `unsteered`
  means the spec was ignored (check the server was started with `--enable-steer-vector
  --steer-algorithms direct` and that the request carried `extra_body["steering"]`).

## Not covered on CPU

Throughput, memory headroom and real numerics (bf16 on H100/H200, Gated-DeltaNet kernels
from flash-linear-attention vs the torch fallback) are only observable on the pod: the
pilot report records generation timings and a GPU-hour estimate, and the replay check
bounds kernel-level numeric differences between vLLM and HF.
