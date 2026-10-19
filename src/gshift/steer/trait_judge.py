"""Open-ended trait-expression judge (held-out Q2 suite, extra_benchmarks.trait_prompts).

Follows the settings recorded in data/trait_openended/PROVENANCE.json
(`judge_settings_followed`), i.e. Betley et al.'s port of the Chen et al. protocol:
  * the frozen rubric (sha256-checked) rendered with trait name, trait definition, the
    BARE bank question (never the natural_* grading prefix) and the visible final answer
    (reasoning removed, never truncated) as the single user message, no system prompt;
  * one judge call per reply, no sampling parameters (Claude Sonnet 5 rejects them),
    adaptive thinking with summarised display only (no effort / budget), a JSON schema
    {score: integer, refusal: boolean} as structured output;
  * strict parse (fences stripped, outermost {...}, score an int in 0..100, refusal a
    bool); an unparseable verdict is recorded as `unparsed`, never re-asked; transport
    errors are retried and recorded as `judge_error` if they persist;
  * empty visible reply -> `no_answer`, never sent; a truncated generation is judged but
    its score is excluded (score None, `truncated`, raw verdict kept); a refusal keeps
    its raw score but `score` is None (reported as a rate beside the mean).
Rows feed extra_benchmarks.trait_summary and steer/report.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from ..io import append_jsonl, read_jsonl
from .extra_benchmarks import RUBRIC_SHA256, fill_template, load_rubric

SCHEMA = {"type": "object", "properties": {"score": {"type": "integer"}, "refusal": {"type": "boolean"}},
          "required": ["score", "refusal"], "additionalProperties": False}
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_verdict(text: str | None) -> tuple[int, bool] | None:
    """Strict parse of the judge's JSON verdict; None if anything is off."""
    if not text:
        return None
    t = _FENCE.sub("", text.strip())
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    s, r = d.get("score"), d.get("refusal")
    if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 100 or not isinstance(r, bool):
        return None
    return s, r


def render(rubric: str, meta: dict, answer: str) -> str:
    return fill_template(rubric, {"trait_name": meta["trait"], "trait_description": meta["trait_description"],
                                  "question": meta["question"], "response": answer}, format_escapes=True)


async def _call(client, cfg: dict, prompt: str) -> tuple[str | None, str | None, str | None, int]:
    """-> (reply text, reasoning summary, error, attempts). Retries transport errors only."""
    extra: dict[str, Any] = {"thinking": {"type": "adaptive", "display": "summarized"}}
    if cfg.get("trait_structured_output", True):
        extra["output_config"] = {"format": {"type": "json_schema", "schema": SCHEMA}}
    err = None
    attempts = int(cfg.get("max_attempts", 4))
    for a in range(1, attempts + 1):
        try:
            r = await client.messages.create(model=cfg["trait_model"], max_tokens=int(cfg.get("trait_max_tokens", 4096)),
                                             messages=[{"role": "user", "content": prompt}], extra_body=extra)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            summ = "".join(getattr(b, "thinking", "") or "" for b in r.content if getattr(b, "type", "") == "thinking")
            return text, (summ or None), None, a
        except Exception as e:  # transport / API errors: back off and retry
            err = repr(e)
            await asyncio.sleep(min(60, 2 ** a + random.random()))
    return None, None, err, attempts


def judge_traits(gen_files: list[Path], out_path: Path, cfg: dict[str, Any]) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env)")
    rubric = load_rubric()
    done = {(r["id"], r["condition"]) for r in read_jsonl(out_path) if not r.get("judge_error")}
    items, direct = [], []
    for f in gen_files:
        for r in read_jsonl(f):
            if (r["id"], r["condition"]) in done:
                continue
            base = {"id": r["id"], "condition": r["condition"], "benchmark": "trait_openended",
                    "trait": r["meta"]["trait"], "judge_model": cfg["trait_model"], "rubric_sha256": RUBRIC_SHA256,
                    "truncated": bool(r.get("truncated"))}
            ans = (r.get("answer_text") or "").strip()
            if not r.get("has_answer") or not ans:
                direct.append({**base, "no_answer": True, "score": None, "refusal": None, "unparsed": False,
                               "judge_error": False})
                continue
            items.append((base, render(rubric, r["meta"], r["answer_text"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if direct:
        append_jsonl(out_path, direct)
    print(f"[trait-judge] {len(items)} replies to judge, {len(direct)} no-answer", flush=True)
    if items:
        asyncio.run(_run(items, cfg, out_path))
    return len(items)


async def _run(items: list[tuple[dict, str]], cfg: dict, out_path: Path) -> None:
    import anthropic

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 16)))
    lock = asyncio.Lock()
    buf: list[dict] = []

    async def one(base: dict, prompt: str):
        async with sem:
            text, summ, err, attempts = await _call(client, cfg, prompt)
        v = parse_verdict(text)
        row = {**base, "no_answer": False, "judge_reply": text, "judge_reasoning": summ, "judge_attempts": attempts,
               "judge_error": err is not None, "unparsed": err is None and v is None,
               "raw_score": v[0] if v else None, "refusal": v[1] if v else None}
        # the score is excluded (None) for refusals, truncated generations and failed verdicts
        row["score"] = v[0] if v and not v[1] and not base["truncated"] else None
        async with lock:
            buf.append(row)
            if len(buf) >= 50:
                append_jsonl(out_path, buf)
                buf.clear()

    await asyncio.gather(*(one(b, p) for b, p in items))
    if buf:
        append_jsonl(out_path, buf)
