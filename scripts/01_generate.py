#!/usr/bin/env python3
"""Stage 1 (GPU, vLLM): generate every request of the given stages for one model.

    python -u scripts/01_generate.py --model qwen3.6-27b --stages main
    python -u scripts/01_generate.py --model qwen3.6-27b --stages wording_validation followup
    python -u scripts/01_generate.py --model qwen3.6-27b --stages pilot --thinking off
    python -u scripts/01_generate.py --model qwen3.6-27b --stages main --smoke
    # 4 GPUs, one process each (roles partitioned):
    CUDA_VISIBLE_DEVICES=1 python -u scripts/01_generate.py --model ... --worker 1 --num-workers 4

Resumable: re-running skips requests already on disk under the same config id, and
refuses to mix outputs generated under a different config. One vLLM engine per process.
"""

import dataclasses

from _common import base_parser, experiment_for, worker_roles

from gshift.config import get_model, require_pinned
from gshift.design import build_requests
from gshift.generate import build_llm, load_tokenizer, run_generation


def main():
    p = base_parser(__doc__)
    p.add_argument("--stages", nargs="+", default=["main"])
    p.add_argument("--roles", nargs="+", default=None, help="restrict to these roles (debugging)")
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=None, help="override the generation budget (smoke tests)")
    p.add_argument("--tensor-parallel-size", type=int, default=None,
                   help="override models.yaml (e.g. 2 on 80 GB GPUs; does not change the config id)")
    args = p.parse_args()
    if "pilot" in args.stages and (len(args.stages) > 1 or not args.thinking):
        p.error("--stages pilot must be run on its own with --thinking on|off (pilot arms go to their own "
                "output roots and never mix with the main grid)")
    if args.thinking and args.stages != ["pilot"] and not args.smoke:
        p.error("--thinking is only for the pilot (or --smoke): the main grid uses generation.enable_thinking")
    exp = experiment_for(args)
    if args.max_tokens:
        exp["generation"]["max_tokens"] = args.max_tokens
    spec = get_model(args.model)
    if args.tensor_parallel_size:
        spec = dataclasses.replace(spec, tensor_parallel_size=args.tensor_parallel_size)
    if not args.smoke:
        require_pinned(spec)
    roles = args.roles or worker_roles(exp, args)
    if args.stages == ["pilot"]:
        reqs = [r for r in build_requests(exp, "pilot") if roles is None or r.role in roles]
    else:
        reqs = [r for st in args.stages for r in build_requests(exp, st, roles)]
    print(f"[generate] {len(reqs)} requests across stages {args.stages} (worker {args.worker}/{args.num_workers})")
    tok = load_tokenizer(spec)
    llm = build_llm(spec, exp["generation"], seed=int(exp["generation"]["base_seed"]))
    run_generation(spec, reqs, exp, chunk_size=args.chunk_size, llm=llm, tokenizer=tok,
                   extra_meta={"stages": args.stages, "smoke": args.smoke})


if __name__ == "__main__":
    main()
