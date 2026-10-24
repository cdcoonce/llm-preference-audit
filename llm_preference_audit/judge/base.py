"""Judge interface: items in presented order, outputs in presented order."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class JudgeItem:
    """One judge call. Texts are already truncated and in PRESENTED order."""

    custom_id: str
    prompt: str
    response_a: str
    response_b: str


@dataclass(frozen=True)
class JudgeOutput:
    """Probabilities are in presented order, normalised when status == "ok", else None."""

    custom_id: str
    status: str  # "ok" | "refused" | "parse_failed" | "errored"
    p_a: float | None
    p_b: float | None
    p_tie: float | None
    winner: str | None
    input_tokens: int
    output_tokens: int


class JudgeAdapter(Protocol):
    """The only interface the harness depends on."""

    name: str
    model: str

    def judge(self, items: Sequence[JudgeItem]) -> list[JudgeOutput]: ...
