#!/usr/bin/env python3
"""Stage A (GPU, HF): ingest completed main-run response files READ-ONLY from the HF dataset,
validate them (PREREG_AB.md A.1), select roles (A.2) and extract all four cross-reading
cells per unit (A.3) into followup_out/A/<model>/. Resumable per role.

    python -u scripts/followup/a_crossread.py --model qwen3.6-27b [--ingest-only]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.config import get_model, load_experiment, role_categories  # noqa: E402
from gshift.followup.crossread import MAIN_CONDS, designed_ids, plan_cells, role_priority, select_roles, validate_file  # noqa: E402
from gshift.generate import config_identity  # noqa: E402

REPO = os.environ.get("HF_DATASET_REPO", "timf34/grader-shifts-persona-space-outputs")


def ingest(exp, spec, cache: Path) -> tuple[dict, dict]:
    """-> (records[role][cond][id], manifest). Only discovery roles in priority order are fetched."""
    from huggingface_hub import HfApi, hf_hub_download

    cid, _ = config_identity(spec, exp)
    files = set(HfApi().list_repo_files(REPO, repo_type="dataset"))
    recs, man = {}, {"repo": REPO, "main_config_id": cid, "roles": {}}
    for cat, order in role_priority(exp).items():
        for role in order:
            entry = {"category": cat, "files": {}, "complete": False}
            want = designed_ids(exp, role)
            rows_by = {}
            ok = True
            for c in MAIN_CONDS:
                f = f"{spec.key}/{c}/responses/{role}.jsonl"
                if f not in files:
                    entry["files"][c] = {"present": False}
                    ok = False
                    continue
                p = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=str(cache))
                raw = Path(p).read_bytes()
                rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
                rows = [r for r in rows if r.get("stage") == "main"]
                why = validate_file(rows, want[c], cid)
                entry["files"][c] = {"present": True, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(rows),
                                     "problems": why}
                ok &= not why
                rows_by[c] = {r["id"]: r for r in rows}
            entry["complete"] = ok
            man["roles"][role] = entry
            if ok:
                recs[role] = rows_by
    return recs, man


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--out", default=str(ROOT / "followup_out" / "A"))
    p.add_argument("--ingest-only", action="store_true")
    args = p.parse_args()
    exp = load_experiment()
    spec = get_model(args.model)
    out = Path(args.out) / spec.key
    out.mkdir(parents=True, exist_ok=True)
    recs, man = ingest(exp, spec, out / "_hf_cache")
    roles, short = select_roles(exp, set(recs))
    man.update(selected_roles=roles, shortfall_by_category=short, ingest_time_utc=time.strftime("%FT%TZ", time.gmtime()))
    cats = role_categories(exp)
    print(f"[A] complete discovery roles: {sorted(recs)}")
    print(f"[A] selected {len(roles)}: {[(r, cats[r]) for r in roles]}; shortfall {short}")
    (out / "ingest_manifest.json").write_text(json.dumps(man, indent=1))
    if args.ingest_only or not roles:
        return
    from gshift.extract import Extractor, load_hf_model
    from gshift.templates import prompt_token_ids

    model, tok = load_hf_model(spec, exp["extraction"].get("dtype", "bfloat16"))
    render = lambda msgs: prompt_token_ids(tok, msgs, True, spec.chat_template_kwargs)  # noqa: E731
    ex = Extractor(model, tok, exp["extraction"]["layer_sweep"], True, spec.chat_template_kwargs)
    problems = {}
    for role in roles:
        dest = out / f"cells_{role}.npz"
        if dest.exists():
            print(f"[A] {role}: done, skipping")
            continue
        jobs, probs = plan_cells(exp, role, recs[role], render)
        problems[role] = probs
        t0 = time.time()
        arrs = {k: [] for k in ("pre", "resp", "reason")}
        norms, index = [], []
        for j in jobs:
            r = ex.record({"prompt_ids": j["prompt_ids"], "gen_ids": j["gen_ids"]})
            for k in arrs:
                arrs[k].append(r["acts"][k].astype(np.float16))
            norms.append(r["norms"])
            sp = r["split"]
            index.append({k: v for k, v in j.items() if k not in ("prompt_ids", "gen_ids")} | {
                "n_prompt": len(j["prompt_ids"]), "n_answer_tokens": len(sp.answer),
                "n_reasoning_tokens": len(sp.reasoning), "has_answer_split": sp.has_answer})
        np.savez(dest, **{k: np.stack(v) for k, v in arrs.items()}, norms=np.stack(norms),
                 layers=np.array(ex.layers), index=np.array([json.dumps(x) for x in index]))
        print(f"[A] {role}: {len(jobs)} cells in {time.time() - t0:.0f}s ({len(probs)} problems)", flush=True)
    man["plan_problems"] = problems
    (out / "ingest_manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
