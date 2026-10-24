"""Shared head/middle/tail truncation applied identically by every LLM judge and model."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
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


def _check_middle_share(middle_share: float) -> None:
    if not 0 <= middle_share < 1:
        raise ValueError(f"middle_share must satisfy 0 <= middle_share < 1, got {middle_share}")


def truncate_text(text: str, budget_tokens: int, middle_share: float = MIDDLE_SHARE) -> Truncated:
    """Keep head/tail verbatim and whole evenly spaced middle sentences within the budget.

    Head and tail each get (1 - middle_share) / 2 of the word budget; the rest is the
    middle-sentence allowance. middle_share=0 keeps head and tail only, with one marker.
    """
    _check_middle_share(middle_share)
    spans = [m.span() for m in _WORD.finditer(text)]
    words_in = len(spans)
    word_budget = int(budget_tokens / TOKENS_PER_WORD)
    if words_in <= word_budget:
        return Truncated(text, False, words_in, words_in, 0)

    edge_share = (1 - middle_share) / 2
    head_n = math.floor(edge_share * word_budget)
    tail_n = math.floor(edge_share * word_budget)
    head = text[: spans[head_n - 1][1]] if head_n else ""
    tail = text[spans[-tail_n][0] :] if tail_n else ""
    middle = text[len(head) : len(text) - len(tail)]

    pieces = [p for p in (s.strip() for s in _SENTENCE_SPLIT.split(middle)) if p]
    counts = [len(p.split()) for p in pieces]
    chosen = set() if middle_share == 0 else set(_select(counts, word_budget - head_n - tail_n))

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


def truncate_turns(
    turns: list[str], budget_tokens: int, middle_share: float = MIDDLE_SHARE
) -> Truncated:
    return truncate_text(conversation_text(turns), budget_tokens, middle_share)


PROMPT_WINDOW_FLOOR = 0.25


def allocate_budgets(len_prompt: int, len_a: int, len_b: int, window: int) -> tuple[int, int, int]:
    """Per-field token allowances (prompt, a, b) within a usable `window`, symmetric in a/b."""
    prompt = min(len_prompt, max(int(PROMPT_WINDOW_FLOOR * window), window - len_a - len_b))
    rest = window - prompt
    if len_a + len_b <= rest:
        return prompt, len_a, len_b
    if min(len_a, len_b) <= rest // 2:
        if len_a <= len_b:
            return prompt, len_a, rest - len_a
        return prompt, rest - len_b, len_b
    return prompt, rest // 2, rest // 2


class FieldFit(NamedTuple):
    truncated: Truncated
    tokens: int
    allowance: int
    fits: bool
    steps: int


def fit_to_allowance(
    turns: list[str],
    allowance: int,
    count: Callable[[str], int],
    middle_share: float = MIDDLE_SHARE,
    step: float = 0.05,
    max_steps: int = 10,
) -> FieldFit:
    """Truncate `turns` so the model tokenizer's `count` fits `allowance`, stepping down."""
    text = conversation_text(turns)
    tokens = count(text)
    words = len(text.split())
    if tokens <= allowance:
        return FieldFit(Truncated(text, False, words, words, 0), tokens, allowance, True, 0)
    ratio = tokens / max(1, words)
    out = truncate_turns(turns, 0, middle_share)
    out_tokens = tokens
    for steps in range(max_steps + 1):
        scale = 1.0 - steps * step
        budget_tokens = int(allowance / ratio * TOKENS_PER_WORD * scale)
        out = truncate_turns(turns, budget_tokens, middle_share)
        out_tokens = count(out.text)
        if out_tokens <= allowance:
            return FieldFit(out, out_tokens, allowance, True, steps)
    return FieldFit(out, out_tokens, allowance, False, max_steps)


class PairFit(NamedTuple):
    prompt: FieldFit
    a: FieldFit
    b: FieldFit
    hard_cut: bool  # some field never fit its allowance; caller must hard-cut at the tokenizer


def fit_pair(
    prompt_turns: list[str],
    a_turns: list[str],
    b_turns: list[str],
    window: int,
    count: Callable[[str], int],
    middle_share: float = MIDDLE_SHARE,
) -> PairFit:
    lens = [count(conversation_text(t)) for t in (prompt_turns, a_turns, b_turns)]
    budgets = allocate_budgets(lens[0], lens[1], lens[2], window)
    prompt, a, b = (
        fit_to_allowance(t, k, count, middle_share)
        for t, k in zip((prompt_turns, a_turns, b_turns), budgets, strict=True)
    )
    return PairFit(prompt, a, b, not (prompt.fits and a.fits and b.fits))


def both_responses_whole(pf: PairFit) -> bool:
    return not pf.a.truncated.truncated and not pf.b.truncated.truncated
