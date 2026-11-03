"""LLM wrapper that works with Groq (free), Claude (Anthropic SDK) or any OpenAI-compatible server."""
from __future__ import annotations

import json
import re

from . import config

_client = None


def get_client():
    global _client
    if _client is None:
        if not config.API_KEY:
            raise RuntimeError(
                f"{config.KEY_ENV_NAME} is not set. Copy backend/.env.example to backend/.env and add your key."
            )
        if config.LLM_PROVIDER == "anthropic":
            from anthropic import Anthropic

            _client = Anthropic(api_key=config.API_KEY)
        else:
            if not config.BASE_URL or not (config.MODEL or "").strip():
                raise RuntimeError("LLM_PROVIDER=custom needs LLM_BASE_URL and LLM_MODEL in backend/.env")
            from openai import OpenAI

            _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL, max_retries=5)
    return _client


def complete(system: str, messages: list[dict], model: str | None = None, max_tokens: int = 1500) -> str:
    model = model or config.MODEL
    client = get_client()

    if config.LLM_PROVIDER == "anthropic":
        resp = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    kwargs = {}
    if "gpt-oss" in model:
        kwargs["extra_body"] = {"reasoning_effort": "low"}
        max_tokens += 800
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, *messages],
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def complete_json(system: str, user: str, model: str | None = None, max_tokens: int = 600) -> dict:
    text = complete(system, [{"role": "user", "content": user}], model=model, max_tokens=max_tokens)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def summarize(text: str) -> str:
    return complete(
        "Summarise the conversation in at most 120 words. Keep names, numbers and open questions.",
        [{"role": "user", "content": text}],
        model=config.FAST_MODEL,
        max_tokens=300,
    )