"""Tests for the generic out-of-fold scorer, on synthetic frames only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import score_oof as so

N = 900


def make_rows(n: int = N, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.arange(n) // 3
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(n)],
            "y": rng.choice(3, n, p=[0.4, 0.4, 0.2]),
            "fold": group % 5,
            "group": [f"g{g}" for g in group],
            "split": "cv",
            "words_a": rng.integers(5, 200, n),
            "words_b": rng.integers(5, 200, n),
        }
    )


def informative(rows: pd.DataFrame, seed: int = 1, strength: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet([4, 4, 4], len(rows)) * (1 - strength)
    P[np.arange(len(rows)), rows["y"].to_numpy()] += strength
    return P / P.sum(axis=1, keepdims=True)


def prior_of(rows: pd.DataFrame) -> np.ndarray:
    return so.prior_by_fold(rows)


def oof_frame(rows: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"id": rows["id"], "p_a": P[:, 0], "p_b": P[:, 1], "p_tie": P[:, 2]})


def test_rejects_missing_id():
    rows = make_rows()
    oof = oof_frame(rows, informative(rows)).iloc[1:]
    with pytest.raises(ValueError, match="1 missing"):
        so.score_oof(oof, rows, prior_of(rows), {})


def test_rejects_duplicate_id():
    rows = make_rows()
    oof = oof_frame(rows, informative(rows))
    oof = pd.concat([oof, oof.iloc[:2]])
    with pytest.raises(ValueError, match="2 duplicate"):
        so.score_oof(oof, rows, prior_of(rows), {})


def test_rejects_test_split_row():
    rows = make_rows()
    rows.loc[3, "split"] = "test"
    with pytest.raises(ValueError, match="1 rows are not in the cv"):
        so.score_oof(oof_frame(rows, informative(rows)), rows, prior_of(rows), {})


def test_rejects_unnormalised_and_negative():
    rows = make_rows()
    P = informative(rows)
    bad = P.copy()
    bad[0] *= 1.1
    with pytest.raises(ValueError, match="do not sum to 1"):
        so.score_oof(oof_frame(rows, bad), rows, prior_of(rows), {})
    neg = P.copy()
    neg[0] = [-0.1, 0.6, 0.5]
    with pytest.raises(ValueError, match="non-negative"):
        so.score_oof(oof_frame(rows, neg), rows, prior_of(rows), {})


def test_informative_model_beats_prior():
    rows = make_rows()
    summary, P_cal = so.score_oof(oof_frame(rows, informative(rows)), rows, prior_of(rows), {})
    assert summary["log_loss"]["matrix_cv"] < summary["log_loss"]["prior_oof"] - 0.3
    assert summary["gain_vs_prior"]["lo"] > 0 and summary["gate_pass"]
    assert P_cal.shape == (N, 3)


def test_identical_comparator_has_zero_diff_and_full_agreement():
    rows = make_rows()
    P = informative(rows)
    summary, _ = so.score_oof(oof_frame(rows, P), rows, prior_of(rows), {"same": P_cal_of(rows, P)})
    c = summary["comparators"]["same"]
    assert c["gain_vs_comparator"]["diff"] == pytest.approx(0, abs=1e-12)
    assert c["pa_minus_pb_correlation"] == pytest.approx(1.0)


def P_cal_of(rows, P):
    return so.score_oof(oof_frame(rows, P), rows, prior_of(rows), {})[1]


def test_better_comparator_gives_negative_diff():
    """Positive = this model better; a stronger comparator makes it negative."""
    rows = make_rows()
    weak = informative(rows, strength=0.2)
    strong = informative(rows, seed=2, strength=0.8)
    summary, _ = so.score_oof(oof_frame(rows, weak), rows, prior_of(rows), {"strong": strong})
    assert summary["comparators"]["strong"]["gain_vs_comparator"]["diff"] < 0


def swap_cols(rows, raw_ab, raw_ba):
    df = oof_frame(rows, (raw_ab + raw_ba[:, [1, 0, 2]]) / 2)
    for o, R in (("ab", raw_ab), ("ba", raw_ba)):
        for k, c in enumerate(("a", "b", "tie")):
            df[f"raw_{o}_{c}"] = R[:, k]
    return df


def test_position_readout_maps_ba_back():
    rows = make_rows()
    P = informative(rows)
    raw_ab = P
    raw_ba = P[:, [1, 0, 2]]  # order-consistent: the same verdict seen from the other slot
    summary, _ = so.score_oof(swap_cols(rows, raw_ab, raw_ba), rows, prior_of(rows), {})
    assert summary["position"]["swap_flip"]["flip_rate"] == 0.0
    # unmapped (wrong) would flip a/b wherever p_a != p_b
    assert summary["position"]["swap_flip"]["n"] == N


def test_first_slot_favouring_model_has_positive_first_position_diff():
    rows = make_rows()
    rng = np.random.default_rng(5)
    first = rng.dirichlet([2, 1, 1], N) * 0 + np.array([0.6, 0.2, 0.2])
    raw_ab, raw_ba = first, first  # always favours whatever is shown first
    df = swap_cols(rows, raw_ab, raw_ba)
    summary, _ = so.score_oof(df, rows, prior_of(rows), {})
    assert summary["position"]["first_position"]["diff"] == pytest.approx(0.4)
    assert summary["position"]["swap_flip"]["flip_rate"] == 1.0


def test_no_position_without_raw_columns():
    rows = make_rows()
    summary, _ = so.score_oof(oof_frame(rows, informative(rows)), rows, prior_of(rows), {})
    assert "position" not in summary


def test_subset_verbosity_uses_identical_rows_for_model_and_comparators():
    rows = make_rows()
    P, Q = informative(rows), informative(rows, seed=3, strength=0.4)
    df = oof_frame(rows, P)
    df["both_untruncated"] = np.arange(N) % 3 != 0
    df["hard_cut"] = np.arange(N) % 10 == 0
    summary, _ = so.score_oof(df, rows, prior_of(rows), {"q": Q})
    for subset in ("both_untruncated", "truncated_any"):
        v = summary["verbosity"][subset]
        assert v["primary"]["n"] == v["q"]["n"] > 0
        assert v["primary"]["human_share"] == v["q"]["human_share"]
    sub = summary["verbosity"]["both_untruncated"]
    assert sub["n_rows"] == int(df["both_untruncated"].sum())
    assert sub["primary"]["n"] != summary["verbosity"]["all"]["primary"]["n"]
    assert summary["counts"]["hard_cut_share"] == pytest.approx(0.1, abs=0.01)
    assert summary["counts"]["both_untruncated_share"] == pytest.approx(2 / 3, abs=0.01)


def test_haiku_columns_are_renamed(tmp_path):
    rows = make_rows()
    P = informative(rows)
    path = tmp_path / "h.parquet"
    pd.DataFrame(
        {
            "id": rows["id"],
            "winner_model_a": P[:, 0],
            "winner_model_b": P[:, 1],
            "winner_tie": P[:, 2],
        }
    ).to_parquet(path)
    assert list(so.read_predictions(path)[so.PROB_COLS].columns) == so.PROB_COLS


def test_cli_ledger_dedupes_by_predictions_sha(tmp_path, monkeypatch):
    rows = make_rows()
    path = tmp_path / "oof.parquet"
    oof_frame(rows, informative(rows)).to_parquet(path)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(so, "LEDGER", ledger)
    monkeypatch.setattr(so, "load_cv_rows", lambda: rows)
    monkeypatch.setattr(so, "load_comparators", lambda r: {})
    monkeypatch.setattr(so, "require_clean", lambda: "abc123")
    argv = [
        "--oof",
        str(path),
        "--name",
        "deb",
        "--model",
        "deberta",
        "--gpu-hours",
        "2.5",
        "--summary",
        str(tmp_path / "s.json"),
        "--ledger",
    ]
    so.main(argv)
    so.main(argv)
    lines = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(lines) == 1
    r = lines[0]
    assert r["config"] == "deb" and r["split"] == "cv-oof" and r["git_sha"] == "abc123"
    assert r["predictions_sha256"] == so.sha256_file(path) and r["gpu_hours"] == 2.5
    assert r["cost_usd"] == 0 and len(r["ci95_vs_prior"]) == 2 and r["n_rows"] == N
    assert (tmp_path / "s.json").exists()


def test_cli_without_ledger_flag_writes_no_row(tmp_path, monkeypatch):
    rows = make_rows()
    path = tmp_path / "oof.parquet"
    oof_frame(rows, informative(rows)).to_parquet(path)
    monkeypatch.setattr(so, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(so, "load_cv_rows", lambda: rows)
    monkeypatch.setattr(so, "load_comparators", lambda r: {})
    so.main(["--oof", str(path), "--name", "x", "--model", "m"])
    assert not (tmp_path / "ledger.jsonl").exists()
