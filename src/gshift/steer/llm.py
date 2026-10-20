"""One async chat call for the Q2 judges, through Anthropic (ANTHROPIC_API_KEY) or, when that
key is absent, OpenRouter (OPENROUTER_API_KEY) with the same Claude model routed to Anthropic
only (no fallbacks). Model names are the Anthropic ids in configs/steering.yaml
(e.g. `claude-sonnet-5`); OpenRouter gets `anthropic/<id>`.

    client = JudgeClient()
    text = await client.chat("claude-sonnet-5", [{"role": "user", "content": prompt}], max_tokens=1024)

Returns the visible text only. Errors propagate (callers retry with their own attempt count).
"""

from __future__ import annotations

import os
from typing import Any


class JudgeClient:
    def __init__(self, provider: str | None = None):
        self.provider = provider or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openrouter")
        if self.provider == "anthropic":
            import anthropic

            self._c = anthropic.AsyncAnthropic()
        elif self.provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("set ANTHROPIC_API_KEY or OPENROUTER_API_KEY (put it in .env)")
            from openai import AsyncOpenAI

            self._c = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=0, timeout=600)
        else:
            raise ValueError(f"unknown judge provider {self.provider!r}")

    async def chat(self, model: str, messages: list[dict[str, str]], max_tokens: int,
                   system: str | None = None, **extra: Any) -> str:
        if self.provider == "anthropic":
            kw: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages, **extra}
            if system:
                kw["system"] = system
            r = await self._c.messages.create(**kw)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        r = await self._c.chat.completions.create(
            model=model if "/" in model else f"anthropic/{model}", messages=msgs, max_tokens=max_tokens,
            extra_body={"provider": {"only": ["anthropic"], "allow_fallbacks": False}, **extra})
        return r.choices[0].message.content or ""
