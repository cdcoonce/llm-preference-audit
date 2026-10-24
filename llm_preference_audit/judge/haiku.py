"""Claude Haiku pairwise judge over the Message Batches API."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Sequence
from typing import Any

from llm_preference_audit.judge.base import JudgeItem, JudgeOutput

MODEL = "claude-haiku-5-5"
EFFORT = "low"
MAX_TOKENS = 2048  # Haiku 5.5 thinks by default and thinking counts toward max_tokens
# Haiku 5.5 is $0.10/$0.50 per MTok for prompts <= 100K tokens; the Batch API is 50% off
# (per the claude-api skill cache, 2026-10-06).
BATCH_INPUT_USD_PER_MTOK = 0.05
BATCH_OUTPUT_USD_PER_MTOK = 0.25

SYSTEM = (
    "You predict human preferences in pairwise chatbot comparisons. You are shown a user "
    "conversation and two anonymous responses, Response A and Response B. Predict which "
    "response a typical human voter on a crowdsourced chatbot-comparison site would prefer, "
    "or whether they would call it a tie. Give calibrated probabilities p_a, p_b and p_tie "
    "that sum to 1, and the single most likely winner (A, B or tie). Long texts may contain "
    "markers like [... N words omitted ...] where a passage was shortened; N is the length "
    "omitted."
)
USER_TEMPLATE = (
    "<prompt>\n{prompt}\n</prompt>\n\n"
    "<response_a>\n{response_a}\n</response_a>\n\n"
    "<response_b>\n{response_b}\n</response_b>"
)
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "p_a": {"type": "number"},
        "p_b": {"type": "number"},
        "p_tie": {"type": "number"},
    },
    "required": ["winner", "p_a", "p_b", "p_tie"],
    "additionalProperties": False,
}
_CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def build_params(item: JudgeItem) -> dict[str, Any]:
    """Message params. No temperature/top_p/top_k (400 since 2026-10-21), no thinking param."""
    content = USER_TEMPLATE.format(
        prompt=item.prompt, response_a=item.response_a, response_b=item.response_b
    )
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def build_request(item: JudgeItem) -> dict[str, Any]:
    if not _CUSTOM_ID.match(item.custom_id):
        raise ValueError(f"invalid custom_id {item.custom_id!r}: need ^[a-zA-Z0-9_-]{{1,64}}$")
    return {"custom_id": item.custom_id, "params": build_params(item)}


def _tokens(message: dict) -> tuple[int, int]:
    usage = message.get("usage") or {}
    inp = sum(
        usage.get(k) or 0
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return int(inp), int(usage.get("output_tokens") or 0)


def _fail(cid: str, status: str, tokens: tuple[int, int]) -> JudgeOutput:
    return JudgeOutput(cid, status, None, None, None, None, tokens[0], tokens[1])


def _parse_text(text: str) -> tuple[float, float, float, str] | None:
    try:
        d = json.loads(text)
        ps = [d["p_a"], d["p_b"], d["p_tie"]]
        winner = d["winner"]
    except (ValueError, KeyError, TypeError):
        return None
    if winner not in ("A", "B", "tie"):
        return None
    if not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in ps):
        return None
    if not all(math.isfinite(p) and p >= 0 for p in ps):
        return None
    total = float(sum(ps))
    if not total > 0:
        return None
    return ps[0] / total, ps[1] / total, ps[2] / total, winner


def parse_result(raw: dict) -> JudgeOutput:
    """Pure: one batch result dict -> JudgeOutput. Never raises."""
    cid = str(raw.get("custom_id", ""))
    result = raw.get("result") or {}
    if result.get("type") != "succeeded":
        return _fail(cid, "errored", (0, 0))
    message = result.get("message") or {}
    tokens = _tokens(message)
    stop = message.get("stop_reason")
    if stop == "refusal":
        return _fail(cid, "refused", tokens)
    if stop == "max_tokens":
        return _fail(cid, "parse_failed", tokens)
    text = next(
        (
            b.get("text")
            for b in message.get("content") or []
            if isinstance(b, dict) and b.get("type") == "text"
        ),
        None,
    )
    parsed = _parse_text(text) if isinstance(text, str) else None
    if parsed is None:
        return _fail(cid, "parse_failed", tokens)
    p_a, p_b, p_tie, winner = parsed
    return JudgeOutput(cid, "ok", p_a, p_b, p_tie, winner, tokens[0], tokens[1])


def cost_usd(outputs: Sequence[JudgeOutput]) -> float:
    inp = sum(o.input_tokens for o in outputs)
    out = sum(o.output_tokens for o in outputs)
    return (inp * BATCH_INPUT_USD_PER_MTOK + out * BATCH_OUTPUT_USD_PER_MTOK) / 1e6


class HaikuBatchJudge:
    """JudgeAdapter backed by the Anthropic Message Batches API."""

    name = "haiku-batch"
    model = MODEL

    def __init__(self, client: Any = None, poll_seconds: float = 30):
        self._client = client
        self.poll_seconds = poll_seconds

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def submit(self, items: Sequence[JudgeItem]) -> str:
        batch = self.client.messages.batches.create(requests=[build_request(i) for i in items])
        return batch.id

    def wait(self, batch_id: str) -> None:
        while self.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(self.poll_seconds)

    def collect(self, batch_id: str) -> list[dict]:
        return [r.model_dump(mode="json") for r in self.client.messages.batches.results(batch_id)]

    def judge(self, items: Sequence[JudgeItem]) -> list[JudgeOutput]:
        batch_id = self.submit(items)
        self.wait(batch_id)
        by_id: dict[str, JudgeOutput] = {}
        for raw in self.collect(batch_id):
            out = parse_result(raw)
            if out.custom_id in by_id:
                raise ValueError(f"duplicate custom_id in batch results: {out.custom_id}")
            by_id[out.custom_id] = out
        missing = [i.custom_id for i in items if i.custom_id not in by_id]
        if missing:
            raise ValueError(f"missing custom_ids in batch results: {missing[:5]}")
        return [by_id[i.custom_id] for i in items]
