#!/usr/bin/env python3
"""Stage B generation (GPU, vLLM; EasySteer venv): 40 tasks x 6 conditions x 3 seeds, default
persona, main-run sampling. Throughput batch first (--probe N), then the rest. Resumable:
records already in the output file are skipped.

    python -u scripts/followup/b_generate.py --model qwen3.6-27b --out followup_out/B --probe 24
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.config import get_model, load_experiment  # noqa: E402
from gshift.followup.shortcut_tasks import CONDITIONS, panel_sha, unit_seed, user_message  # noqa: E402

N_SEEDS = 3


def requests(panel: list[dict]) -> list[dict]:
    out = []
    for t in panel:
        for k in range(N_SEEDS):
            seed = unit_seed(t["task_id"], k)
            for c in CONDITIONS:
                out.append({"id": f"{t['task_id']}|{c}|s{k}", "task_id": t["task_id"], "split": t["split"],
                            "condition": c, "seed_idx": k, "seed": seed,
                            "messages": [{"role": "user", "content": user_message(t, c)}]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--out", default=str(ROOT / "followup_out" / "B"))
    p.add_argument("--probe", type=int, default=24, help="throughput batch size before the rest")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    exp = load_experiment()
    spec = get_model(args.model)
    gen = exp["generation"]
    panel = [json.loads(l) for l in open(ROOT / "data" / "followup" / "shortcut_panel_v1.jsonl")]
    reqs = requests(panel)[: args.limit]
    out_dir = Path(args.out) / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "generations.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    todo = [r for r in reqs if r["id"] not in done]
    print(f"[B-gen] {len(reqs)} requests, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    from vllm.inputs import TokensPrompt

    from gshift.generate import build_llm, load_tokenizer, sampling_params
    from gshift.templates import answer_text, inspect_template, prompt_token_ids, reasoning_text, split_generation

    tok = load_tokenizer(spec)
    info = inspect_template(tok, True, spec.chat_template_kwargs)
    llm = build_llm(spec, gen, seed=int(gen["base_seed"]))
    max_len = int(gen["max_model_len"])
    meta = {"model": spec.hf_id, "revision": spec.revision, "panel_sha256": panel_sha(panel),
            "sampling": {k: gen[k] for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens")},
            "template": info.to_dict()}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    timings = []
    batches = [todo[: args.probe], todo[args.probe:]] if args.probe else [todo]
    for b in batches:
        if not b:
            continue
        prompts, params, pids = [], [], []
        for r in b:
            ids = prompt_token_ids(tok, r["messages"], True, spec.chat_template_kwargs)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(sampling_params(gen, r["seed"], min(int(gen["max_tokens"]), max_len - len(ids))))
            pids.append(ids)
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=True)
        dt = time.time() - t0
        n_tok = 0
        with open(path, "a") as f:
            for r, ids, o in zip(b, pids, outs):
                c = o.outputs[0]
                g = list(c.token_ids)
                n_tok += len(g)
                sp = split_generation(g, tok, info)
                f.write(json.dumps({**r, "prompt_ids": ids, "gen_ids": g, "finish_reason": c.finish_reason,
                                    "truncated": c.finish_reason == "length",
                                    "answer_text": answer_text(g, tok, sp), "reasoning_text": reasoning_text(g, tok, sp),
                                    **sp.to_dict()}) + "\n")
        timings.append({"n": len(b), "seconds": round(dt, 1), "gen_tokens": n_tok, "tok_per_s": round(n_tok / dt, 1)})
        print(f"[B-gen] batch {len(b)}: {dt:.0f}s, {n_tok} tokens, {n_tok / dt:.0f} tok/s", flush=True)
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
