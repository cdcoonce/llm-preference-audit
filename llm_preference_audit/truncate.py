"""Shared head/middle/tail truncation applied identically by every LLM judge and model."""

from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import NamedTuple

from llm_preference_audit.data import conversation_text
from llm_preference_audit.profile import TOKENS_PER_WORD

HEAD_SHARE = 0.4
MIDDLE_SHARE = 0.2
TAIL_SHARE = 0.4
PROMPT_BUDGET_TOKENS = 1000
RESPONSE_BUDGET_TOKENS = 1500
MARKER_WORDS = 5

_WORD = re.compile(r"\S+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


class Truncated(NamedTuple):
    text: str
    truncated: bool
    words_in: int
    words_kept: int  # excludes marker words
    words_omitted: int


def omission_marker(n: int) -> str:
    return f"[... {n} words omitted ...]"


def _select(counts: list[int], budget: int) -> list[int]:
    """Indices at the smallest fixed centred cadence whose cost fits `budget`."""
    n = len(counts)
    for c in range(1, n + 1):
        idx = list(range((c - 1) // 2, n, c))
        gaps = (idx[0] > 0) + (idx[-1] < n - 1) + sum(b - a > 1 for a, b in pairwise(idx))
        if sum(counts[i] for i in idx) + MARKER_WORDS * gaps <= budget:
            return idx
    return []


def truncate_text(text: str, budget_tokens: int) -> Truncated:
    """Keep head/tail verbatim and whole evenly spaced middle sentences within the budget."""
    spans = [m.span() for m in _WORD.finditer(text)]
    words_in = len(spans)
    word_budget = int(budget_tokens / TOKENS_PER_WORD)
    if words_in <= word_budget:
        return Truncated(text, False, words_in, words_in, 0)

    head_n = math.floor(HEAD_SHARE * word_budget)
    tail_n = math.floor(TAIL_SHARE * word_budget)
    head = text[: spans[head_n - 1][1]] if head_n else ""
    tail = text[spans[-tail_n][0] :] if tail_n else ""
    middle = text[len(head) : len(text) - len(tail)]

    pieces = [p for p in (s.strip() for s in _SENTENCE_SPLIT.split(middle)) if p]
    counts = [len(p.split()) for p in pieces]
    chosen = set(_select(counts, word_budget - head_n - tail_n))

    parts = [head] if head else []
    kept = head_n + tail_n
    run = 0
    for i, piece in enumerate(pieces):
        if i in chosen:
            if run:
                parts.append(omission_marker(run))
                run = 0
            parts.append(piece)
            kept += counts[i]
        else:
            run += counts[i]
    if run:
        parts.append(omission_marker(run))
    if tail:
        parts.append(tail)
    return Truncated("\n".join(parts), True, words_in, kept, words_in - kept)


def truncate_turns(turns: list[str], budget_tokens: int) -> Truncated:
    return truncate_text(conversation_text(turns), budget_tokens)
