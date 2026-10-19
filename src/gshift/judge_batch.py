"""Role-adherence judging through OpenRouter's Batch API: the same judge model, prompt and
parse rule as gshift.judge (Assistant Axis rubric, 0-3), billed at ~50% and completed
asynchronously within 24 h. Writes the same scores/<role>.jsonl rows (the analysis reads
the last row per id), so live and batch judging are interchangeable.

Scope: by default only MAIN-grid records (wording_set discovery / none): the adherence
analyses never read the wording-validation or follow-up stages.

Flow (resumable; state in <outputs>/<model>/judge_batch/state.json):
  submit  build prompts for every unscored record, POST them in chunks to /api/v1/batches
  poll    fetch every pending batch; when complete, append score rows
  rounds  unparseable verdicts are resubmitted (up to max_rounds), like the live judge's
          re-asks; still-unparseable ones keep score=None and are reported
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .io import append_jsonl, model_dir, read_jsonl, responses_path, scores_path, write_json
from .judge import build_judge_prompt, parse_score

BATCH_URL = "https://openrouter.ai/api/v1/batches"
TERMINAL = ("completed", "failed", "expired", "cancelled")


def _headers() -> dict[str, str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def state_path(model_key: str) -> Path:
    return model_dir(model_key) / "judge_batch" / "state.json"


def load_state(model_key: str) -> dict[str, Any]:
    p = state_path(model_key)
    return json.loads(p.read_text()) if p.exists() else {"batches": {}, "items": {}}


def collect_items(model_key: str, pairs: list[tuple[str, str]], default_role: str,
                  wording_sets: tuple[str, ...] = ("discovery", "none"),
                  exclude_ids: set[str] | None = None) -> tuple[list[dict], int]:
    """Unscored records to judge (same rules as gshift.judge.run_judge); writes the
    default-role / no-answer skip rows directly. Returns (items, n_skipped)."""
    exclude_ids = exclude_ids or set()
    items, n_skip = [], 0
    for cond, role in pairs:
        out = scores_path(model_key, cond, role)
        done = {r["id"] for r in read_jsonl(out) if r.get("score") is not None or r.get("skipped")}
        skipped = []
        for rec in read_jsonl(responses_path(model_key, cond, role)):
            if rec["id"] in done or rec["id"] in exclude_ids or rec.get("wording_set", "none") not in wording_sets:
                continue
            base = {"id": rec["id"], "role": role, "condition": cond, "question_id": rec["question_id"]}
            if role == default_role:
                skipped.append({**base, "score": None, "skipped": "default_role"})
            elif not rec.get("has_answer"):
                skipped.append({**base, "score": None, "skipped": "no_answer"})
            else:
                # ORIGINAL question (no prefix, no condition label) + final answer only
                items.append({**base, "out": str(out),
                              "prompt": build_judge_prompt(role, rec["question"], rec["answer_text"])})
        if skipped:
            out.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(out, skipped)
            n_skip += len(skipped)
    return items, n_skip


def submit(model_key: str, items: list[dict], cfg: dict[str, Any], chunk: int = 2000, round_: int = 1,
           session=None) -> list[str]:
    """POST items in chunks; records custom_id -> item in the state file. Returns batch ids."""
    import requests

    session = session or requests
    st = load_state(model_key)
    model = cfg["model"].split(":")[0]
    ids = []
    for start in range(0, len(items), chunk):
        part = items[start:start + chunk]
        reqs = []
        for k, it in enumerate(part):
            cid = f"r{round_}-{len(st['items'])}"
            st["items"][cid] = {k2: it[k2] for k2 in ("id", "role", "condition", "question_id", "out", "prompt")} \
                | {"round": round_}
            body: dict[str, Any] = {"messages": [{"role": "user", "content": it["prompt"]}],
                                    "max_tokens": int(cfg.get("max_tokens", 10))}
            if cfg.get("temperature") is not None:
                body["temperature"] = float(cfg["temperature"])
            reqs.append({"custom_id": cid, "body": body})
        payload: dict[str, Any] = {"endpoint": "/v1/chat/completions", "model": model}   # order matters
        if cfg.get("route"):
            payload["provider"] = {"only": [r.lower() for r in cfg["route"]], "allow_fallbacks": False}
        payload["completion_window"] = "24h"
        payload["requests"] = reqs
        r = session.post(BATCH_URL, headers=_headers(), data=json.dumps(payload), timeout=300)
        if r.status_code not in (200, 202):
            raise RuntimeError(f"batch submit failed: HTTP {r.status_code}: {r.text[:500]}")
        bid = r.json()["id"]
        st["batches"][bid] = {"status": "validating", "n": len(reqs), "round": round_,
                              "custom_ids": [q["custom_id"] for q in reqs], "submitted": time.time()}
        ids.append(bid)
        write_json(state_path(model_key), st)       # after every chunk: a crash never loses a batch id
        print(f"[judge-batch] submitted {bid}: {len(reqs)} requests (round {round_})", flush=True)
    return ids


def poll(model_key: str, cfg: dict[str, Any], session=None) -> dict[str, int]:
    """Fetch every non-terminal batch; write score rows for completed ones."""
    import requests

    session = session or requests
    st = load_state(model_key)
    counts = {"pending": 0, "completed": 0, "failed": 0, "rows": 0, "unparsed": 0}
    for bid, b in st["batches"].items():
        if b.get("written"):
            counts["completed"] += 1
            continue
        r = session.get(f"{BATCH_URL}/{bid}", headers=_headers(), timeout=300)
        if r.status_code != 200:
            print(f"[judge-batch] {bid}: HTTP {r.status_code} {r.text[:200]}", flush=True)
            counts["pending"] += 1
            continue
        d = r.json()
        b["status"] = d.get("status")
        b["request_counts"] = d.get("request_counts")
        if b["status"] not in TERMINAL:
            counts["pending"] += 1
            continue
        if b["status"] != "completed" and not d.get("results"):
            counts["failed"] += 1
            b["written"] = True
            b["error"] = str(d.get("errors") or d.get("error"))[:500]
            continue
        by_file: dict[str, list[dict]] = {}
        for res in d.get("results") or []:
            it = st["items"].get(res.get("custom_id"))
            if it is None:
                continue
            reply, err = None, None
            if res.get("response"):
                try:
                    reply = res["response"]["body"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    err = "malformed response"
            else:
                err = json.dumps(res.get("error"))[:300]
            score = parse_score(reply)
            row = {"id": it["id"], "role": it["role"], "condition": it["condition"], "question_id": it["question_id"],
                   "score": score, "reply": reply, "error": err, "judge_provider": "openrouter-batch",
                   "judge_model": cfg["model"], "batch_id": bid, "round": it["round"]}
            by_file.setdefault(it["out"], []).append(row)
            counts["rows"] += 1
            counts["unparsed"] += score is None
        for path, rows in by_file.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(path, rows)
        b["written"] = True
        counts["completed"] += 1
        write_json(state_path(model_key), st)
    write_json(state_path(model_key), st)
    return counts


def resubmit_unparsed(model_key: str, cfg: dict[str, Any], max_rounds: int = 3, chunk: int = 2000,
                      session=None) -> int:
    """Items whose latest score row is unparseable (and below max_rounds) go into a new round."""
    st = load_state(model_key)
    if any(not b.get("written") for b in st["batches"].values()):
        return 0
    latest: dict[str, dict] = {}
    for path in {it["out"] for it in st["items"].values()}:
        for r in read_jsonl(path):
            latest[r["id"]] = r
    by_id = {}
    for it in st["items"].values():
        by_id[it["id"]] = it if it["round"] >= by_id.get(it["id"], {"round": 0})["round"] else by_id[it["id"]]
    redo = [it for i, it in by_id.items() if latest.get(i, {}).get("score") is None
            and not latest.get(i, {}).get("skipped") and it["round"] < max_rounds]
    if not redo:
        return 0
    nxt = max(it["round"] for it in redo) + 1
    submit(model_key, redo, cfg, chunk=chunk, round_=nxt, session=session)
    return len(redo)
