import json
import math
import random
from pathlib import Path

import pytest
from anthropic.types.messages import MessageBatchIndividualResponse

from llm_preference_audit.judge.base import JudgeItem
from llm_preference_audit.judge.haiku import (
    MODEL,
    HaikuBatchJudge,
    build_params,
    build_request,
    cost_usd,
    parse_result,
)

FIXTURE = Path(__file__).parent / "fixtures" / "haiku_batch_results.jsonl"


def _load() -> dict[str, dict]:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]
    return {r["custom_id"]: r for r in rows}


RAW = _load()


def test_fixtures_match_sdk_shape():
    for raw in RAW.values():
        MessageBatchIndividualResponse.model_validate(raw)


def test_parse_ok():
    o = parse_result(RAW["ok-1"])
    assert o.status == "ok" and o.winner == "A"
    assert (o.p_a, o.p_b, o.p_tie) == pytest.approx((0.6, 0.3, 0.1))
    assert (o.input_tokens, o.output_tokens) == (120, 30)


def test_parse_normalises():
    o = parse_result(RAW["ok-unnorm"])
    assert o.status == "ok"
    assert o.p_a + o.p_b + o.p_tie == pytest.approx(1.0)
    assert o.p_b == pytest.approx(0.6 / 0.9)


def test_parse_skips_thinking_and_counts_cache_tokens():
    o = parse_result(RAW["ok-thinking"])
    assert o.status == "ok" and o.winner == "tie"
    assert o.input_tokens == 115 and o.output_tokens == 50


def test_parse_refusal():
    o = parse_result(RAW["refused"])
    assert o.status == "refused" and o.p_a is None and o.winner is None


@pytest.mark.parametrize("cid", ["max-tokens", "bad-json", "neg-p", "bad-winner"])
def test_parse_failures(cid):
    o = parse_result(RAW[cid])
    assert o.status == "parse_failed" and o.p_a is None and o.winner is None


@pytest.mark.parametrize("cid", ["errored", "expired"])
def test_parse_errored(cid):
    o = parse_result(RAW[cid])
    assert o.status == "errored" and o.input_tokens == 0 and o.output_tokens == 0


def test_parse_never_raises_on_odd_shapes():
    for raw in (
        {"custom_id": "x", "result": {"type": "succeeded", "message": {"content": []}}},
        {"custom_id": "x", "result": {"type": "canceled"}},
        {
            "custom_id": "x",
            "result": {
                "type": "succeeded",
                "message": {"content": [{"type": "text", "text": "[1,2]"}]},
            },
        },
    ):
        assert parse_result(raw).status in {"parse_failed", "errored"}


def test_parse_rejects_nonfinite_and_zero_sum():
    def raw(text):
        return {
            "custom_id": "x",
            "result": {
                "type": "succeeded",
                "message": {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]},
            },
        }

    assert parse_result(raw('{"winner":"A","p_a":0,"p_b":0,"p_tie":0}')).status == "parse_failed"
    assert parse_result(raw('{"winner":"A","p_a":NaN,"p_b":1,"p_tie":0}')).status == "parse_failed"
    assert parse_result(raw('{"winner":"A","p_a":1,"p_b":0}')).status == "parse_failed"
    overflow = '{"winner":"A","p_a":1e308,"p_b":1e308,"p_tie":0}'
    assert parse_result(raw(overflow)).status == "parse_failed"


ITEM = JudgeItem("r1-ab", "PROMPT_TXT", "FIRST_RESP", "SECOND_RESP")


def test_build_params():
    p = build_params(ITEM)
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in p
    assert p["model"] == MODEL == "claude-haiku-5-5"
    assert p["output_config"]["effort"] == "low"
    fmt = p["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == {"winner", "p_a", "p_b", "p_tie"}
    assert "minimum" not in json.dumps(fmt["schema"]) and "maximum" not in json.dumps(fmt["schema"])
    assert isinstance(p["system"], str) and p["max_tokens"] == 2048


def test_build_params_message_order():
    (m,) = build_params(ITEM)["messages"]
    assert m["role"] == "user"
    c = m["content"]
    assert c.index("PROMPT_TXT") < c.index("FIRST_RESP") < c.index("SECOND_RESP")
    assert c.index("<response_a>") < c.index("FIRST_RESP") < c.index("</response_a>")
    assert c.index("<response_b>") < c.index("SECOND_RESP") < c.index("</response_b>")


def test_build_request_and_custom_id_validation():
    r = build_request(ITEM)
    assert r["custom_id"] == "r1-ab" and r["params"] == build_params(ITEM)
    for bad in ("", "a b", "x" * 65, "a.b"):
        with pytest.raises(ValueError):
            build_request(JudgeItem(bad, "p", "a", "b"))


def test_cost_usd():
    outs = [parse_result(RAW["ok-1"]), parse_result(RAW["ok-thinking"])]
    expected = (235 * 0.05 + 80 * 0.25) / 1e6
    assert cost_usd(outs) == pytest.approx(expected)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeBatches:
    def __init__(self, results, polls_before_end=1):
        self._results = results
        self._polls = polls_before_end
        self.created = None
        self.retrieved = 0

    def create(self, requests):
        self.created = requests
        return _Obj(id="batch_1", processing_status="in_progress")

    def retrieve(self, batch_id):
        assert batch_id == "batch_1"
        self.retrieved += 1
        done = self.retrieved > self._polls
        return _Obj(processing_status="ended" if done else "in_progress")

    def results(self, batch_id):
        assert batch_id == "batch_1"
        return [MessageBatchIndividualResponse.model_validate(r) for r in self._results]


class FakeClient:
    def __init__(self, results, **kw):
        self.messages = _Obj(batches=FakeBatches(results, **kw))
        self.options = []

    def with_options(self, **kw):
        self.options.append(kw)
        return self


def _judge(results, monkeypatch, **kw):
    sleeps = []
    monkeypatch.setattr("llm_preference_audit.judge.haiku.time.sleep", sleeps.append)
    client = FakeClient(results, **kw)
    return HaikuBatchJudge(client=client, poll_seconds=7), client, sleeps


def test_judge_orders_by_custom_id(monkeypatch):
    ids = ["ok-1", "ok-unnorm", "refused", "bad-json", "expired"]
    items = [JudgeItem(i, "p", "a", "b") for i in ids]
    shuffled = [RAW[i] for i in ids]
    random.Random(3).shuffle(shuffled)
    assert [r["custom_id"] for r in shuffled] != ids
    j, client, sleeps = _judge(shuffled, monkeypatch)
    outs = j.judge(items)
    assert [o.custom_id for o in outs] == ids
    assert [o.status for o in outs] == ["ok", "ok", "refused", "parse_failed", "errored"]
    assert [r["custom_id"] for r in client.messages.batches.created] == ids
    assert sleeps == [7]


def test_judge_missing_or_duplicate_raises(monkeypatch):
    items = [JudgeItem(i, "p", "a", "b") for i in ("ok-1", "ok-unnorm")]
    j, _, _ = _judge([RAW["ok-1"]], monkeypatch)
    with pytest.raises(ValueError, match="ok-unnorm"):
        j.judge(items)
    j, _, _ = _judge([RAW["ok-1"], RAW["ok-1"], RAW["ok-unnorm"]], monkeypatch)
    with pytest.raises(ValueError, match="ok-1"):
        j.judge(items)


def test_constructor_makes_no_client():
    j = HaikuBatchJudge()
    assert j._client is None and j.name and j.model == MODEL
    assert math.isfinite(j.poll_seconds)


RECORDED = Path(__file__).parent / "fixtures" / "haiku_batch_results_recorded.jsonl"


def test_recorded_pilot_responses_parse_ok():
    """Real Batch API results from the 2026-10-24 pilot (ids renamed; JSON answers only)."""
    rows = [json.loads(line) for line in RECORDED.read_text().splitlines() if line]
    assert len(rows) == 6
    for raw in rows:
        MessageBatchIndividualResponse.model_validate(raw)
        out = parse_result(raw)
        assert out.status == "ok" and out.input_tokens > 0 and out.output_tokens > 0
        assert math.isclose(out.p_a + out.p_b + out.p_tie, 1.0)


def test_submit_disables_sdk_retries_on_create(monkeypatch):
    """A retried create after a read timeout could bill a second, unrecorded batch."""
    j, client, _ = _judge([RAW["ok-1"]], monkeypatch)
    j.submit([JudgeItem("ok-1", "p", "a", "b")])
    assert client.options == [{"max_retries": 0}]
