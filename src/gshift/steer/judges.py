"""School of Reward Hacks judges (Betley et al.'s port; prompts verbatim, same model for both).

Judge 1 (metric) sees the ORIGINAL prompt with the scoring rule and scores how literally the
answer satisfies the rule; judge 2 (quality) sees the DE-SCORED prompt and scores quality.
gap = metric - quality per sample. A reply with no visible answer (truncated / empty) is
`no_answer` and is never sent to a judge; an unparseable judge reply is `unparsed`, never
turned into a number. Only the visible final answer is judged (no reasoning).
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from pathlib import Path
from typing import Any

from .. import DATA_DIR
from ..io import append_jsonl, read_jsonl

_SCORE = re.compile(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>")


def _template(name: str) -> str:
    return (DATA_DIR / "school_of_reward_hacks" / name).read_text()


def parse_last_score(text: str | None) -> float | None:
    if not text:
        return None
    m = _SCORE.findall(text)
    if not m:
        return None
    v = float(m[-1])
    return v if 0 <= v <= 100 else None


async def _judge(client, model: str, prompt: str, max_tokens: int, attempts: int) -> tuple[float | None, str | None, int]:
    reply = None
    for a in range(1, attempts + 1):
        try:
            reply = await client.chat(model, [{"role": "user", "content": prompt}], max_tokens)
            s = parse_last_score(reply)
            if s is not None:
                return s, reply, a
        except Exception as e:
            reply = f"ERROR: {e!r}"
            await asyncio.sleep(min(60, 2 ** a + random.random()))
    return None, reply, attempts


async def _run(items: list[dict], cfg: dict, out_path: Path) -> None:
    from .llm import JudgeClient

    client = JudgeClient()
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 16)))
    t1, t2 = _template("judge1_metric_prompt.txt"), _template("judge2_quality_prompt.txt")
    lock = asyncio.Lock()
    buf: list[dict] = []

    async def one(it):
        async with sem:
            row = {"id": it["id"], "condition": it["condition"], "benchmark": it["benchmark"],
                   "judge_model": cfg["sorh_model"]}
            ans = it["answer"]
            if it["benchmark"] == "sorh":
                p1 = t1.replace("{prompt}", it["user"]).replace("{answer}", ans)
                row["metric"], row["metric_reply"], row["metric_attempts"] = await _judge(
                    client, cfg["sorh_model"], p1, int(cfg["max_tokens"]), int(cfg["max_attempts"]))
            p2 = t2.replace("{prompt}", it["user_descored"]).replace("{answer}", ans)
            row["quality"], row["quality_reply"], row["quality_attempts"] = await _judge(
                client, cfg["sorh_model"], p2, int(cfg["max_tokens"]), int(cfg["max_attempts"]))
            if it["benchmark"] == "sorh":
                ok = row["metric"] is not None and row["quality"] is not None
                row["gap"] = row["metric"] - row["quality"] if ok else None
                row["unparsed"] = not ok
            else:
                row["unparsed"] = row["quality"] is None
            async with lock:
                buf.append(row)
                if len(buf) >= 50:
                    append_jsonl(out_path, buf)
                    buf.clear()

    await asyncio.gather(*(one(it) for it in items))
    if buf:
        append_jsonl(out_path, buf)


def judge_sorh(gen_files: list[Path], out_path: Path, cfg: dict[str, Any]) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        raise RuntimeError("set ANTHROPIC_API_KEY or OPENROUTER_API_KEY (put it in .env)")
    done = {(r.get("benchmark"), r["id"], r["condition"]) for r in read_jsonl(out_path) if not r.get("unparsed")}
    items, no_answer = [], []
    for f in gen_files:
        for r in read_jsonl(f):
            if (r["benchmark"], r["id"], r["condition"]) in done:
                continue
            base = {"id": r["id"], "condition": r["condition"], "benchmark": r["benchmark"]}
            if not r.get("has_answer") or not r.get("answer_text", "").strip():
                no_answer.append({**base, "no_answer": True, "metric": None, "quality": None, "gap": None})
                continue
            items.append({**base, "answer": r["answer_text"], "user": r["meta"]["user"],
                          "user_descored": r["meta"]["user_descored"]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if no_answer:
        append_jsonl(out_path, no_answer)
    print(f"[sorh-judge] {len(items)} answers to judge, {len(no_answer)} no-answer", flush=True)
    if items:
        asyncio.run(_run(items, cfg, out_path))
    return len(items)
