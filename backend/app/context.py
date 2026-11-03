"""Context engineering: deciding *what* goes into the model's context window.

Techniques implemented here:
  1. Near-duplicate removal  - overlapping chunks waste tokens
  2. Token-budget packing    - best chunks first, stop at the budget
  3. Citation formatting     - numbered sources the model can cite as [n]
  4. History compaction      - old turns are summarised, recent turns stay verbatim
"""
from __future__ import annotations

from typing import Callable

from . import config


def estimate_tokens(text: str) -> int:
    """Cheap approximation (~4 chars per token); good enough for budgeting."""
    return max(1, len(text) // 4)


def _words(text: str) -> set[str]:
    return set(text.lower().split())


def dedupe(chunks: list[dict], threshold: float = 0.8) -> list[dict]:
    """Drop chunks whose word-set Jaccard similarity to an already-kept chunk is high."""
    kept: list[dict] = []
    kept_words: list[set[str]] = []
    for ch in sorted(chunks, key=lambda c: c["score"], reverse=True):
        w = _words(ch["text"])
        duplicate = False
        for kw in kept_words:
            union = len(w | kw) or 1
            if len(w & kw) / union >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(ch)
            kept_words.append(w)
    return kept


def pack(chunks: list[dict], budget: int = config.CONTEXT_TOKEN_BUDGET) -> tuple[list[dict], int]:
    """Greedily keep highest-scoring chunks that fit in the token budget."""
    packed: list[dict] = []
    used = 0
    for ch in sorted(chunks, key=lambda c: c["score"], reverse=True):
        cost = estimate_tokens(ch["text"])
        if used + cost > budget:
            continue
        packed.append(ch)
        used += cost
    return packed, used


def format_context(chunks: list[dict]) -> str:
    """Number each chunk so the model can cite it as [1], [2], ..."""
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({ch['source']}, page {ch['page']})\n{ch['text']}")
    return "\n\n".join(blocks)


def compact_history(
    history: list[dict],
    summarize: Callable[[str], str],
    keep_turns: int = config.HISTORY_KEEP_TURNS,
) -> list[dict]:
    """Keep the last `keep_turns` exchanges verbatim; fold older ones into a summary.

    `history` is a list of {"role": "user"|"assistant", "content": str}.
    """
    keep = keep_turns * 2
    if len(history) <= keep:
        return history
    old, recent = history[:-keep], history[-keep:]
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old)
    summary = summarize(transcript)
    return [
        {"role": "user", "content": f"(Summary of earlier conversation: {summary})"},
        {"role": "assistant", "content": "Understood, I'll keep that context in mind."},
        *recent,
    ]
