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
    args = p.parse_args()
    exp = experiment_for(args)
    if args.max_tokens:
        exp["generation"]["max_tokens"] = args.max_tokens
    spec = get_model(args.model)
    if not args.smoke:
        require_pinned(spec)
    roles = args.roles or worker_roles(exp, args)
    reqs = [r for st in args.stages for r in build_requests(exp, st, roles)] if "pilot" not in args.stages else \
        [r for r in build_requests(exp, "pilot") if roles is None or r.role in roles]
    print(f"[generate] {len(reqs)} requests across stages {args.stages} (worker {args.worker}/{args.num_workers})")
    tok = load_tokenizer(spec)
    llm = build_llm(spec, exp["generation"], seed=int(exp["generation"]["base_seed"]))
    run_generation(spec, reqs, exp, chunk_size=args.chunk_size, llm=llm, tokenizer=tok,
                   extra_meta={"stages": args.stages, "smoke": args.smoke})


if __name__ == "__main__":
    main()
