import json
from argparse import Namespace

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
    (work / "submit.lock").write_text("t")
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
    assert rows[0]["cost_usd"] == pytest.approx((1000 * 0.05 + 200 * 0.25) * len(raws) / 1e6)
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["status_counts"]["ok"] == len(raws) and summary["gate_pass"] is True
