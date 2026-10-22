#!/usr/bin/env python3
"""Q2 stage B report (CPU): persona choice (decisions vs prose) and objective vs proxy.

    python -u scripts/18_stage_b_report.py --model qwen3.6-27b
-> outputs/<model>/steering/stage_b_report.json
"""

import json

from _common import base_parser, experiment_for

from gshift.io import read_jsonl, write_json
from gshift.steer.vectors import steering_dir


def gen_rows(sd, prefix):
    rows = []
    for d in sorted((sd / "gen").glob(prefix + "*")):
        for f in sorted(d.glob("*.jsonl")):
            rows += read_jsonl(f)
    return rows


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    experiment_for(args)
    sd = steering_dir(args.model)
    out = {"model": args.model}
    ch = gen_rows(sd, "choice_")
    if ch:
        from gshift.steer.persona_choice import choice_rows, choice_summary

        out["persona_choice"] = choice_summary(choice_rows(ch))
        j = sd / "choice_adherence_judgements.jsonl"
        if j.exists():
            from gshift.steer.adherence import adherence_summary

            out["persona_choice_prose_adherence"] = adherence_summary(j)
    px = gen_rows(sd, "proxy_")
    if px:
        from gshift.steer.proxy_panel import proxy_rows, proxy_summary

        out["proxy"] = proxy_summary(proxy_rows(px))
    ac = sd / "activation_check.json"
    if ac.exists():
        a = json.loads(ac.read_text())
        out["convergence"] = {k: {c: {kk: e[kk] for kk in ("beta_crossfit", "beta_ci95", "separation_ratio", "n_roles")}
                                  for c, e in v.get("convergence", {}).items()} for k, v in a["pre"].items()}
        out["convergence_vs_random"] = a.get("convergence_vs_random")
    write_json(sd / "stage_b_report.json", out)
    print(json.dumps({k: (list(v)[:8] if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
