import hashlib
import json
from argparse import Namespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import run_haiku_pilot as pilot_mod
from run_haiku_pilot import score

from llm_preference_audit.judge.base import JudgeOutput

PRIOR = np.array([0.35, 0.34, 0.31])


def _pilot(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(1000, 1000 + n),
            "y": rng.choice(3, size=n, p=PRIOR),
            "fold": np.arange(n) % 5,
            "group": [f"g{i // 2}" for i in range(n)],  # pairs of rows share a prompt group
            "split": "cv",
            "words_a": rng.integers(10, 400, size=n),
            "words_b": rng.integers(10, 400, size=n),
        }
    )


def _out(cid: str, p: tuple[float, float, float], status: str = "ok") -> JudgeOutput:
    if status != "ok":
        return JudgeOutput(cid, status, None, None, None, None, 100, 0)
    return JudgeOutput(cid, "ok", *p, "A", 1000, 200)


def _judge(pilot: pd.DataFrame, sharp: float, refuse: set[str] = frozenset()) -> list:
    """Fake judge: puts `sharp` mass on the true winner, in PRESENTED order for each call."""
    outs = []
    for i, y in zip(pilot["id"], pilot["y"]):
        orig = np.full(3, (1 - sharp) / 2)
        orig[y] = sharp
        for order, p in (("ab", orig), ("ba", orig[[1, 0, 2]])):
            cid = f"{i}-{order}"
            outs.append(_out(cid, tuple(p), "refused" if cid in refuse else "ok"))
    return outs


def test_informative_judge_passes_gate_through_swap_mapping():
    pilot = _pilot()
    summary, P_cal = score(pilot, np.tile(PRIOR, (len(pilot), 1)), _judge(pilot, 0.7))
    assert summary["gate_pass"] is True
    assert summary["log_loss"]["matrix_cv"] < summary["log_loss"]["prior_oof"]
    # A ba call mapped back wrongly would put mass on the loser and flip every row.
    assert summary["position"]["swap_flip"]["flip_rate"] == 0
    assert (P_cal.argmax(axis=1) == pilot["y"].to_numpy()).mean() > 0.9


def test_uninformative_judge_fails_gate():
    pilot = _pilot()
    outs = [_out(o.custom_id, (1 / 3, 1 / 3, 1 / 3)) for o in _judge(pilot, 0.7)]
    summary, _ = score(pilot, np.tile(PRIOR, (len(pilot), 1)), outs)
    assert summary["gate_pass"] is False


def test_refusals_counted_and_excluded_view_reported():
    pilot = _pilot()
    refuse = {f"{pilot['id'][0]}-ab", f"{pilot['id'][1]}-ba", f"{pilot['id'][1]}-ab"}
    summary, _ = score(pilot, np.tile(PRIOR, (len(pilot), 1)), _judge(pilot, 0.7, refuse))
    assert summary["status_counts"]["refused"] == 3
    assert summary["refusal_rate"] == pytest.approx(3 / (2 * len(pilot)))
    assert summary["fallback_row_share"] == pytest.approx(2 / len(pilot))
    assert summary["log_loss"]["matrix_cv_excluding_fallback_rows"] is not None


def test_test_split_row_is_refused_before_calibration():
    pilot = _pilot()
    pilot.loc[5, "split"] = "test"
    with pytest.raises(ValueError):
        score(pilot, np.tile(PRIOR, (len(pilot), 1)), _judge(pilot, 0.7))


def test_cost_comes_from_reported_usage():
    pilot = _pilot(n=50)
    summary, _ = score(pilot, np.tile(PRIOR, (len(pilot), 1)), _judge(pilot, 0.7))
    calls = 2 * len(pilot)
    assert summary["tokens"] == {"input": 1000 * calls, "output": 200 * calls}
    assert summary["cost_usd"] == pytest.approx((1000 * 0.05 + 200 * 0.25) * calls / 1e6)


def test_refused_calls_do_not_count_as_swap_flips():
    pilot = _pilot(n=100)
    refuse = {f"{i}-ab" for i in pilot["id"][:50]}
    summary, _ = score(pilot, np.tile(PRIOR, (len(pilot), 1)), _judge(pilot, 0.7, refuse))
    flip = summary["position"]["swap_flip"]
    assert flip["n"] == 50
    assert flip["flip_rate"] == 0


def test_winner_argmax_agreement_reads_tie():
    pilot = _pilot(n=60)
    outs = [
        JudgeOutput(o.custom_id, "ok", o.p_a, o.p_b, o.p_tie, ("A", "B", "tie")[k % 3], 1, 1)
        for k, o in enumerate(_judge(pilot, 0.7))
    ]
    summary, _ = score(pilot, np.tile(PRIOR, (len(pilot), 1)), outs)
    expected = np.mean(
        [o.winner == ("A", "B", "tie")[int(np.argmax([o.p_a, o.p_b, o.p_tie]))] for o in outs]
    )
    assert summary["winner_matches_argmax_share"] == pytest.approx(expected)
    tie_ok = [o for o in outs if o.winner == "tie" and o.p_tie == max(o.p_a, o.p_b, o.p_tie)]
    assert tie_ok and expected > 0


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Point every stage at tmp paths and synthetic rows; no real data, git or network."""
    pilot = _pilot(n=60)
    pilot["prompt"] = [["synthetic question"]] * len(pilot)
    pilot["response_a"] = [["short synthetic answer."]] * len(pilot)
    pilot["response_b"] = [["another synthetic answer."]] * len(pilot)
    monkeypatch.setattr(pilot_mod, "WORK", tmp_path / "work")
    monkeypatch.setattr(pilot_mod, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(pilot_mod, "SUMMARY", tmp_path / "summary.json")
    monkeypatch.setattr(pilot_mod, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(pilot_mod, "require_clean", lambda: "deadbeef")
    monkeypatch.setattr(pilot_mod, "load_pilot", lambda: (pilot, None))
    monkeypatch.setattr(pilot_mod, "load_train", lambda _p: pilot.assign(label="a"))
    monkeypatch.setattr(
        pilot_mod, "prior_by_fold", lambda _m, _l, folds: np.tile(PRIOR, (len(folds), 1))
    )
    (tmp_path / "work").mkdir()
    return pilot, tmp_path


def _raw(o: JudgeOutput) -> dict:
    text = json.dumps({"winner": "A", "p_a": o.p_a, "p_b": o.p_b, "p_tie": o.p_tie})
    message = {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }
    return {"custom_id": o.custom_id, "result": {"type": "succeeded", "message": message}}


def test_count_stage_runs_and_records_request_digest(staged, monkeypatch):
    pilot, tmp = staged

    class FakeMessages:
        def count_tokens(self, **kw):
            assert "temperature" not in kw and kw["output_config"]["effort"] == "low"
            return Namespace(input_tokens=321)

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: Namespace(messages=FakeMessages()))
    pilot_mod.stage_count(Namespace())
    record = json.loads((tmp / "work" / "token_counts.json").read_text())
    assert len(record["counts"]) == 2 * len(pilot)
    items, _ = pilot_mod.build_items(pilot)
    assert record["requests_sha256"] == pilot_mod.requests_sha256(items)


def test_submit_refuses_stale_counts_and_leftover_lock(staged):
    _, tmp = staged
    work = tmp / "work"
    (work / "token_counts.json").write_text(json.dumps({"requests_sha256": "x", "counts": {}}))
    with pytest.raises(SystemExit, match="re-run `count`"):
        pilot_mod.stage_submit(Namespace(yes=True))
    (work / "submit-0.lock").write_text("t")
    with pytest.raises(SystemExit, match="Console"):
        pilot_mod.stage_submit(Namespace(yes=True))


def test_collect_stage_scores_and_appends_one_ledger_row(staged):
    pilot, tmp = staged
    (tmp / "work" / "batch_id.txt").write_text("msgbatch_test\n")
    raws = [_raw(o) for o in _judge(pilot, 0.7)]
    (tmp / "work" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raws))
    pilot_mod.stage_collect(Namespace())
    pilot_mod.stage_collect(Namespace())  # re-run must not double-count spend
    rows = [json.loads(x) for x in (tmp / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["batch_id"] == "msgbatch_test"
    assert rows[0]["batch_ids"] == ["msgbatch_test"]
    assert rows[0]["cost_usd"] == pytest.approx((1000 * 0.05 + 200 * 0.25) * len(raws) / 1e6)
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["status_counts"]["ok"] == len(raws) and summary["gate_pass"] is True


def test_collect_rerun_leaves_recorded_summary_untouched(staged):
    _, tmp = staged
    (tmp / "work" / "batch_id.txt").write_text("msgbatch_test\n")
    raws = [_raw(o) for o in _judge(staged[0], 0.7)]
    (tmp / "work" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raws))
    pilot_mod.stage_collect(Namespace())
    (tmp / "summary.json").write_text("RECORDED\n")
    pilot_mod.stage_collect(Namespace())
    assert (tmp / "summary.json").read_text() == "RECORDED\n"


def test_collect_refuses_ledger_row_when_too_many_calls_failed(staged):
    pilot, tmp = staged
    (tmp / "work" / "batch_id.txt").write_text("msgbatch_test\n")
    raws = [_raw(o) for o in _judge(pilot, 0.7)]
    for r in raws[:5]:  # 5 of 120 calls expired: 4.2% > 2%
        r["result"] = {"type": "expired"}
    (tmp / "work" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raws))
    with pytest.raises(SystemExit, match="not ok"):
        pilot_mod.stage_collect(Namespace())
    assert not (tmp / "ledger.jsonl").exists()
    assert json.loads((tmp / "summary.json").read_text())["status_counts"]["errored"] == 5


def test_count_refuses_a_tiny_sample(staged, monkeypatch):
    import anthropic

    def no_network():
        raise AssertionError("count reached the API client")

    monkeypatch.setattr(anthropic, "Anthropic", no_network)
    with pytest.raises(SystemExit, match="below"):
        pilot_mod.stage_count(Namespace(sample=50))


# ---------------------------------------------------------------- full-run extensions


def _varied(pilot: pd.DataFrame) -> pd.DataFrame:
    """Rows of different lengths so ratio estimators have something to average."""
    out = pilot.copy()
    out["response_a"] = [["word " * (3 + 7 * (k % 9))] for k in range(len(out))]
    return out


def _ids(chunk) -> list[str]:
    return [i.custom_id for i in chunk]


class FakeJudge:
    """Records create calls; optionally dies on the n-th create (simulated crash)."""

    created: ClassVar[list[list[str]]] = []
    die_on: ClassVar[int | None] = None
    waited: ClassVar[list[str]] = []
    results: ClassVar[dict[str, list[dict]]] = {}

    def __init__(self):
        pass

    def submit(self, items):
        cls = type(self)
        if cls.die_on is not None and len(cls.created) == cls.die_on:
            raise RuntimeError("network died mid-create")
        cls.created.append(_ids(items))
        return f"msgbatch_{len(cls.created) - 1}_{_ids(items)[0]}"

    def wait(self, batch_id):
        type(self).waited.append(batch_id)

    def collect(self, batch_id):
        return type(self).results[batch_id]


@pytest.fixture
def judge(monkeypatch):
    FakeJudge.created, FakeJudge.die_on, FakeJudge.waited, FakeJudge.results = [], None, [], {}
    monkeypatch.setattr(pilot_mod, "HaikuBatchJudge", FakeJudge)
    return FakeJudge


def _write_counts(pilot, work, per_call=1000, p99=None):
    items, _ = pilot_mod.build_items(pilot)
    total = per_call * len(items)
    record = {
        "requests_sha256": pilot_mod.requests_sha256(items),
        "sampled": False,
        "n_counted": len(items),
        "counts": {i.custom_id: per_call for i in items},
        "projected_input_tokens": total,
        "projected_input_tokens_p99": p99 if p99 is not None else total,
    }
    (work / "token_counts.json").write_text(json.dumps(record))
    return items


def test_chunking_keeps_ab_ba_together_and_respects_cap(monkeypatch):
    from llm_preference_audit.judge.harness import make_items

    n = 25
    items = make_items(
        [str(i) for i in range(n)], ["p"] * n, ["a"] * n, ["b"] * n
    )  # ab, ba per row
    for cap, rows_per in ((10, 5), (7, 3), (1, 1)):
        monkeypatch.setattr(pilot_mod, "CHUNK_REQUESTS", cap)
        chunks = pilot_mod.chunk_items(items)
        assert [i for c in chunks for i in c] == items
        for c in chunks:
            assert len(c) <= max(cap, 2) and len(c) // 2 <= rows_per
            ids = _ids(c)
            assert {x.rsplit("-", 1)[0] for x in ids if x.endswith("-ab")} == {
                x.rsplit("-", 1)[0] for x in ids if x.endswith("-ba")
            }
        assert len(chunks) == -(-n // rows_per)


def test_oversize_chunk_refused_before_any_create(staged, judge, monkeypatch):
    pilot, tmp = staged
    _write_counts(pilot, tmp / "work")
    monkeypatch.setattr(pilot_mod, "MAX_CHUNK_BYTES", 100)
    with pytest.raises(SystemExit, match="bytes"):
        pilot_mod.stage_submit(Namespace(yes=True))
    assert judge.created == []


def test_submit_resumes_only_missing_chunks(staged, judge, monkeypatch):
    pilot, tmp = staged
    work = tmp / "work"
    _write_counts(pilot, work)
    monkeypatch.setattr(pilot_mod, "CHUNK_REQUESTS", 40)  # 20 rows -> 3 chunks of 60 rows
    judge.die_on = 1  # chunk 0 is created, chunk 1 dies mid-create
    with pytest.raises(RuntimeError):
        pilot_mod.stage_submit(Namespace(yes=True))
    state = json.loads((work / "batches.json").read_text())
    assert list(state["chunks"]) == ["0"] and state["n_chunks"] == 3
    assert (work / "submit-1.lock").exists() and not (work / "submit-0.lock").exists()
    with pytest.raises(SystemExit, match="chunk 1"):
        pilot_mod.stage_submit(Namespace(yes=True))  # leftover lock: check the Console first
    (work / "submit-1.lock").unlink()
    judge.die_on = None
    pilot_mod.stage_submit(Namespace(yes=True))
    assert len(judge.created) == 3  # chunk 0 once, then chunks 1 and 2 only
    assert len({tuple(c) for c in judge.created}) == 3
    state = json.loads((work / "batches.json").read_text())
    assert sorted(state["chunks"]) == ["0", "1", "2"]
    assert not list(work.glob("submit-*.lock"))
    pilot_mod.stage_submit(Namespace(yes=True))  # all done: nothing more is created
    assert len(judge.created) == 3


def test_pilot_back_compat_existing_batch_id_is_not_resubmitted(staged, judge):
    pilot, tmp = staged
    work = tmp / "work"
    _write_counts(pilot, work)
    (work / "batch_id.txt").write_text("msgbatch_old\n")
    pilot_mod.stage_submit(Namespace(yes=True))
    assert judge.created == []
    assert pilot_mod.load_batches(None)["chunks"] == {"0": "msgbatch_old"}


def test_submit_without_yes_creates_nothing(staged, judge):
    pilot, tmp = staged
    _write_counts(pilot, tmp / "work")
    with pytest.raises(SystemExit, match="--yes"):
        pilot_mod.stage_submit(Namespace(yes=False))
    assert judge.created == []


def test_sampled_count_uses_ratio_estimator(staged, monkeypatch):
    pilot, tmp = staged
    pilot = _varied(pilot)
    monkeypatch.setattr(pilot_mod, "load_pilot", lambda: (pilot, None))
    items, _ = pilot_mod.build_items(pilot)
    local = {pilot_mod.build_params(i)["messages"][0]["content"]: i for i in items}
    local_total = sum(pilot_mod.local_tokens(i) for i in items)

    class FakeMessages:
        def count_tokens(self, **kw):
            return Namespace(
                input_tokens=2 * pilot_mod.local_tokens(local[kw["messages"][0]["content"]])
            )

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: Namespace(messages=FakeMessages()))
    monkeypatch.setattr(pilot_mod, "MIN_COUNT_SAMPLE", 1)  # 60-row synthetic frame
    pilot_mod.stage_count(Namespace(sample=20))
    rec = json.loads((tmp / "work" / "token_counts.json").read_text())
    assert rec["sampled"] is True and rec["n_counted"] == 40 == len(rec["counts"])
    rows = {k.rsplit("-", 1)[0] for k in rec["counts"]}
    assert len(rows) == 20 and all(
        f"{r}-ab" in rec["counts"] and f"{r}-ba" in rec["counts"] for r in rows
    )
    assert rec["projected_input_tokens"] == pytest.approx(2 * local_total, rel=1e-6)
    assert rec["projected_input_tokens_p99"] >= rec["projected_input_tokens"] * (1 - 1e-9)
    assert rec["requests_sha256"] == pilot_mod.requests_sha256(items)


def test_sampled_count_is_deterministic(staged, monkeypatch):
    ids = [str(i) for i in range(500)]
    a = pilot_mod.count_sample_ids(ids, 50)
    assert a == pilot_mod.count_sample_ids(list(reversed(ids)), 50) and len(a) == 50
    assert pilot_mod.count_sample_ids(ids, None) == set(ids)


def test_old_token_counts_format_is_read_as_exact():
    old = {"requests_sha256": "x", "counts": {"a-ab": 5, "a-ba": 7}}
    assert pilot_mod.projected_tokens(old) == (12, 12)


def test_gate_refuses_over_allotment_and_ceiling():
    assert pilot_mod.check_gate(estimate=0.5, spent=10.0, ceiling=1.0) is None
    with pytest.raises(SystemExit, match="allotment"):
        pilot_mod.check_gate(estimate=5.0, spent=36.0, ceiling=None)
    with pytest.raises(SystemExit, match="ceiling"):
        pilot_mod.check_gate(estimate=1.5, spent=0.0, ceiling=1.0)
    assert pilot_mod.check_gate(estimate=5.0, spent=34.0, ceiling=None) is None


def test_submit_gate_reads_ledger_spend(staged, judge, monkeypatch):
    pilot, tmp = staged
    _write_counts(pilot, tmp / "work", p99=10_000_000)  # about $0.5 + output
    monkeypatch.setitem(pilot_mod.RUNS["pilot-500"], "ceiling_usd", None)
    (tmp / "ledger.jsonl").write_text(json.dumps({"cost_usd": 39.9}) + "\n")
    with pytest.raises(SystemExit, match="allotment"):
        pilot_mod.stage_submit(Namespace(yes=True))
    assert judge.created == []
    (tmp / "ledger.jsonl").write_text(json.dumps({"cost_usd": 1.0}) + "\n")
    pilot_mod.stage_submit(Namespace(yes=True))
    assert len(judge.created) == 1


def test_output_rate_uses_measured_pilot_with_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot_mod, "PILOT_SUMMARY", tmp_path / "none.json")
    assert pilot_mod.output_tokens_per_call() == 400
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"n_calls": 1000, "tokens": {"output": 70080}}))
    monkeypatch.setattr(pilot_mod, "PILOT_SUMMARY", f)
    assert pilot_mod.output_tokens_per_call() == pytest.approx(70.08 * 2.0)


def test_collect_over_chunks_scores_and_appends_one_ledger_row(staged, judge, monkeypatch):
    pilot, tmp = staged
    work = tmp / "work"
    items = _write_counts(pilot, work)
    monkeypatch.setattr(pilot_mod, "CHUNK_REQUESTS", 40)
    pilot_mod.stage_submit(Namespace(yes=True))
    state = json.loads((work / "batches.json").read_text())
    chunks = pilot_mod.chunk_items(items)
    by_id = {o.custom_id: _raw(o) for o in _judge(pilot, 0.7)}
    rng = np.random.default_rng(1)
    for k, chunk in enumerate(chunks):
        raws = [by_id[i] for i in _ids(chunk)]
        judge.results[state["chunks"][str(k)]] = [raws[j] for j in rng.permutation(len(raws))]
    pilot_mod.stage_collect(Namespace())
    first_waits = list(judge.waited)
    pilot_mod.stage_collect(Namespace())  # raw files exist; ledger already has this batch set
    rows = [json.loads(x) for x in (tmp / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["batch_ids"] == sorted(state["chunks"].values()) and len(first_waits) == 3
    assert rows[0]["n_rows"] == len(pilot)
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["n_chunks"] == 3 and summary["batch_ids"] == rows[0]["batch_ids"]
    assert summary["status_counts"]["ok"] == len(items) and summary["gate_pass"] is True
    assert summary["position"]["swap_flip"]["flip_rate"] == 0
    names = sorted(p.name for p in work.glob("results*.jsonl"))
    assert names == [
        f"results-1-{state['chunks']['1']}.jsonl",
        f"results-2-{state['chunks']['2']}.jsonl",
        "results.jsonl",
    ]


def test_collect_refuses_when_a_chunk_was_never_submitted(staged, judge):
    _pilot_rows, tmp = staged
    (tmp / "work" / "batches.json").write_text(
        json.dumps({"chunks": {"0": "msgbatch_a"}, "n_chunks": 2, "requests_sha256": "x"})
    )
    with pytest.raises(SystemExit, match="submit"):
        pilot_mod.stage_collect(Namespace())


def test_config_hash_differs_between_runs(monkeypatch):
    pilot_hash = pilot_mod.config_hash()
    monkeypatch.setattr(pilot_mod, "RUN", "full-cv")
    assert pilot_mod.config_hash() != pilot_hash


def test_run_settings_and_select_run(monkeypatch):
    full = pilot_mod.RUNS["full-cv"]
    assert full["n_rows"] is None and full["ceiling_usd"] is None and full["default_sample"] == 2000
    assert str(full["summary"]) == "results/rung-3-full.json"
    assert full["ledger_config"] == "haiku_full_cv" and full["split_tag"] == "cv-full"
    pilot_cfg = pilot_mod.RUNS["pilot-500"]
    assert pilot_cfg["n_rows"] == 500 and pilot_cfg["ceiling_usd"] == 1.0
    monkeypatch.setattr(pilot_mod, "WORK", pilot_mod.WORK)
    monkeypatch.setattr(pilot_mod, "SUMMARY", pilot_mod.SUMMARY)
    monkeypatch.setattr(pilot_mod, "RUN", pilot_mod.RUN)
    pilot_mod.select_run("full-cv")
    assert pilot_mod.WORK == full["work"] and pilot_mod.RUN == "full-cv"


def test_pilot_sample_ids_unchanged():
    frame = pd.DataFrame({"id": np.arange(0, 5000, 3), "label": "winner_tie"})
    expected = sorted(
        sorted(
            frame["id"],
            key=lambda i: hashlib.sha256(f"haiku-pilot-2026-10-23:{i}".encode()).hexdigest(),
        )[:500]
    )
    assert pilot_mod.sample_rows(frame, 500)["id"].tolist() == expected
    assert pilot_mod.sample_rows(frame, None)["id"].tolist() == sorted(frame["id"])
