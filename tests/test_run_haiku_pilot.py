import numpy as np
import pandas as pd
import pytest
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
