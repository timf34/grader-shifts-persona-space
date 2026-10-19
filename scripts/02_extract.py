#!/usr/bin/env python3
"""Stage 2 (GPU, HF transformers): activation summaries + Betley reproduction/verification.

    python -u scripts/02_extract.py --model qwen3.6-27b
    CUDA_VISIBLE_DEVICES=1 python -u scripts/02_extract.py --model qwen3.6-27b --worker 1 --num-workers 4 --skip-betley

Order: (1) reproduce Betley et al.'s automated-vs-human vector with OUR extractor on THEIR
540 prompts (all 64 layers) and, on Qwen3.6-27B, check it against the published arrays --
the layer-indexing / position / chat-template verification (--require-verification stops
the run if it fails); (2) extract every (condition, role) response file on disk at the
pre-registered layer sweep. Resumable per (condition, role).
"""

import json
import sys

from _common import base_parser, experiment_for, worker_roles

from gshift.config import get_model, require_pinned
from gshift.extract import load_hf_model, run_extraction
from gshift.io import conditions_present, model_dir, roles_present


def main():
    p = base_parser(__doc__)
    p.add_argument("--conditions", nargs="+", default=None, help="default: every condition on disk")
    p.add_argument("--layers", nargs="+", type=int, default=None, help="override the layer sweep")
    p.add_argument("--skip-betley", action="store_true")
    p.add_argument("--betley-only", action="store_true")
    p.add_argument("--require-verification", action="store_true")
    args = p.parse_args()
    exp = experiment_for(args)
    spec = get_model(args.model)
    if not args.smoke:
        require_pinned(spec)
    layers = args.layers or exp["extraction"]["layer_sweep"]
    enable_thinking = bool(exp["generation"]["enable_thinking"])
    model, tok = load_hf_model(spec, exp["extraction"].get("dtype", "bfloat16"))
    if not args.skip_betley:
        from gshift.betley import reproduce

        out = model_dir(spec.key) / "betley_repro"
        if (out / "report.json").exists():
            rep = json.loads((out / "report.json").read_text())
            print("[betley] reproduction exists; reusing")
        else:
            rep = reproduce(model, tok, out, enable_thinking, spec.chat_template_kwargs)
        if not spec.key.startswith("qwen3.6"):
            # the published vector is Qwen3.6-27B's: for another model it is a cross-model
            # reference (expected to differ), not a verification of our conventions
            rep["comparison_with_qwen3.6_published"] = {k: rep.get(k) for k in (
                "cos_with_published_at_primary", "norm_ratio_at_primary", "best_offset_by_norm_profile")}
            rep["verification_passed"] = None
            rep["verification_note"] = "published vector is Qwen3.6-27B's; not applicable to " + spec.key
            (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        keys = ["cos_with_published_at_primary", "norm_ratio_at_primary", "best_offset_by_norm_profile",
                "templated_prompt_matches_published", "verification_passed", "comparison"]
        print("[betley] " + json.dumps({k: rep.get(k) for k in keys}), flush=True)
        if args.require_verification and rep.get("verification_passed") is False:
            print("!!! Betley verification FAILED: layer/position/template conventions do not match the "
                  "published vector. Stopping before extraction. See betley_repro/report.json.", flush=True)
            sys.exit(2)
    if args.betley_only:
        return
    roles = set(worker_roles(exp, args) or []) or None
    pairs = []
    for cond in args.conditions or conditions_present(spec.key):
        for role in roles_present(spec.key, cond):
            if roles is None or role in roles:
                pairs.append((cond, role))
    print(f"[extract] {len(pairs)} (condition, role) files for worker {args.worker}/{args.num_workers}")
    run_extraction(model, tok, spec.key, pairs, layers, enable_thinking, spec.chat_template_kwargs,
                   save_dtype=exp["extraction"].get("save_dtype", "float16"),
                   extra_meta={"model": spec.hf_id, "revision": spec.revision})


if __name__ == "__main__":
    main()
