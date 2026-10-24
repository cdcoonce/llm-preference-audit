"""Pair each row in both orders, run a judge, swap-average back to original order."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from llm_preference_audit.judge.base import JudgeItem, JudgeOutput


def make_items(
    ids: Sequence[str],
    prompts: Sequence[str],
    responses_a: Sequence[str],
    responses_b: Sequence[str],
) -> list[JudgeItem]:
    """Two items per row: `{id}-ab` (A=response_a) then `{id}-ba` (A=response_b)."""
    items: list[JudgeItem] = []
    for i, p, a, b in zip(ids, prompts, responses_a, responses_b, strict=True):
        items.append(JudgeItem(f"{i}-ab", p, a, b))
        items.append(JudgeItem(f"{i}-ba", p, b, a))
    return items


def _index(outputs: Sequence[JudgeOutput]) -> dict[str, JudgeOutput]:
    by_id: dict[str, JudgeOutput] = {}
    for o in outputs:
        if o.custom_id in by_id:
            raise ValueError(f"duplicate custom_id: {o.custom_id}")
        by_id[o.custom_id] = o
    return by_id


def combine(outputs: list[JudgeOutput], ids: Sequence[str], prior) -> dict:
    """Swap-average; `prior` is (a, b, tie) in original order and replaces any non-ok call."""
    prior = np.asarray(prior, dtype=float)
    by_id = _index(outputs)
    n = len(ids)
    raw = {s: np.full((n, 3), np.nan) for s in ("ab", "ba")}
    mapped = {s: np.tile(prior, (n, 1)) for s in ("ab", "ba")}
    status = {s: np.empty(n, dtype=object) for s in ("ab", "ba")}
    for r, i in enumerate(ids):
        for s in ("ab", "ba"):
            o = by_id.get(f"{i}-{s}")
            if o is None:
                raise ValueError(f"missing custom_id: {i}-{s}")
            status[s][r] = o.status
            if o.status != "ok":
                continue
            presented = np.array([o.p_a, o.p_b, o.p_tie], dtype=float)
            raw[s][r] = presented
            mapped[s][r] = presented if s == "ab" else presented[[1, 0, 2]]
    return {
        "P": (mapped["ab"] + mapped["ba"]) / 2,
        "P_ab": mapped["ab"],
        "P_ba": mapped["ba"],
        "raw_ab": raw["ab"],
        "raw_ba": raw["ba"],
        "status_ab": status["ab"],
        "status_ba": status["ba"],
        "fallback": (status["ab"] != "ok") | (status["ba"] != "ok"),
    }


def status_counts(outputs: Sequence[JudgeOutput]) -> dict[str, int]:
    return dict(Counter(o.status for o in outputs))


def refusal_rate(outputs: Sequence[JudgeOutput]) -> float:
    return sum(o.status == "refused" for o in outputs) / len(outputs) if outputs else 0.0
