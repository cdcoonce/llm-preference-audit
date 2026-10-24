import numpy as np
import pytest

from llm_preference_audit.bias import identical_rows_readout, longer_wins_readout


def test_longer_wins_readout():
    y = np.array([0, 1, 2, 0, 1])
    wa = np.array([10, 10, 5, 3, 7])
    wb = np.array([5, 20, 9, 3, 2])  # row 2 is a tie label, row 3 has equal lengths
    P = np.array(
        [[0.6, 0.2, 0.2], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]]
    )
    r = longer_wins_readout(y, P, wa, wb)
    assert r["n"] == 3
    assert r["human_longer_share"] == pytest.approx(2 / 3)  # row 4: b won but a is longer
    assert r["direction_agreement"] == pytest.approx(1.0)


def test_identical_rows_readout():
    P = np.array([[0.4, 0.4, 0.2], [0.5, 0.3, 0.2], [0.3, 0.5, 0.2]])
    r = identical_rows_readout(P, np.array([True, False, False]))
    assert r["n"] == 1 and r["share_favours_a"] == 0.0 and r["max_abs_p_a_minus_p_b"] == 0.0
    assert identical_rows_readout(P, np.zeros(3, bool))["n"] == 0


def _longer_frame(n_groups=2000, p_human=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * 2
    wa = rng.integers(10, 200, n)
    wb = rng.integers(10, 200, n)
    wb = np.where(wb == wa, wb + 1, wb)
    longer_is_a = wa > wb
    longer_wins = rng.random(n) < p_human
    y = np.where(longer_wins == longer_is_a, 0, 1)
    groups = np.repeat(np.arange(n_groups), 2)
    return y, wa, wb, longer_is_a, groups


def _probs(longer_is_a, p_longer):
    p_l = np.full(len(longer_is_a), p_longer)
    pa = np.where(longer_is_a, p_l, 1 - p_l) * 0.9
    pb = np.where(longer_is_a, 1 - p_l, p_l) * 0.9
    return np.stack([pa, pb, np.full(len(pa), 0.1)], axis=1)


def test_calibrated_predictor_has_no_excess_length_lean():
    from llm_preference_audit.bias import calibrated_longer_readout

    y, wa, wb, la, g = _longer_frame(p_human=0.6)
    r = calibrated_longer_readout(y, _probs(la, 0.6), wa, wb, g)
    assert r["human_share"] == pytest.approx(0.6, abs=0.03)
    assert r["model_share"] == pytest.approx(0.6, abs=1e-9)
    assert abs(r["diff"]) < 0.03 and r["ci95"][0] < 0 < r["ci95"][1]


def test_overconfident_predictor_has_positive_diff():
    from llm_preference_audit.bias import calibrated_longer_readout

    y, wa, wb, la, g = _longer_frame(p_human=0.6)
    r = calibrated_longer_readout(y, _probs(la, 0.9), wa, wb, g)
    assert r["model_share"] == pytest.approx(0.9, abs=1e-9)
    assert r["diff"] > 0.25 and r["ci95"][0] > 0
    assert sum(b["n"] for b in r["buckets"]) == r["n"] and len(r["buckets"]) == 5


def test_calibrated_readout_handles_empty_responses():
    from llm_preference_audit.bias import calibrated_longer_readout

    y, wa, wb, la, g = _longer_frame(p_human=0.6)
    wa = wa.copy()
    wa[:50] = 0  # empty response a
    y = np.where(wa > wb, 0, 1)
    la = wa > wb
    r = calibrated_longer_readout(y, _probs(la, 0.6), wa, wb, g)
    assert all(np.isfinite([b["lo"] for b in r["buckets"]] + [b["hi"] for b in r["buckets"]]))


def test_swap_flip_readout_hand_example_excludes_nan_rows():
    from llm_preference_audit.bias import swap_flip_readout

    nan = np.full(3, np.nan)
    P_ab = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], nan, [0.1, 0.1, 0.8]])
    P_ba = np.array([[0.2, 0.6, 0.2], [0.2, 0.4, 0.4], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]])
    r = swap_flip_readout(P_ab, P_ba)
    assert r["n"] == 3
    assert r["flip_rate"] == pytest.approx(1 / 3)  # only row 0 flips
    assert r["mean_abs_diff"] == pytest.approx((0.4 + 0.1 + 0.0) / 3)


def test_swap_flip_readout_excludes_row_with_nan_in_either_order():
    from llm_preference_audit.bias import swap_flip_readout

    ok = np.array([[0.5, 0.3, 0.2]])
    nan = np.full((1, 3), np.nan)
    assert swap_flip_readout(np.vstack([ok, ok]), np.vstack([ok, nan]))["n"] == 1


def test_first_position_readout_hand_example():
    from llm_preference_audit.bias import first_position_readout

    nan = np.full(3, np.nan)
    ab = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], nan])
    ba = np.array([[0.5, 0.4, 0.1], [0.3, 0.5, 0.2], [0.4, 0.45, 0.15]])
    r = first_position_readout(ab, ba, np.array([0, 1, 2]))
    assert r["n_calls"] == 5
    assert r["mean_p_first"] == pytest.approx((0.6 + 0.2 + 0.5 + 0.3 + 0.4) / 5)
    assert r["mean_p_second"] == pytest.approx((0.3 + 0.2 + 0.4 + 0.5 + 0.45) / 5)
    assert r["diff"] == pytest.approx(r["mean_p_first"] - r["mean_p_second"])
    # the 0.6-tie call is dropped; first wins in 2 of the 4 non-tie calls
    assert r["share_first_wins_argmax_nontie"] == pytest.approx(2 / 4)


def test_first_position_readout_detects_first_position_bias():
    from llm_preference_audit.bias import first_position_readout

    n = 400
    raw = np.tile([0.6, 0.3, 0.1], (n, 1))
    r = first_position_readout(raw, raw, np.arange(n) // 2)
    assert r["diff"] == pytest.approx(0.3)
    assert r["ci95"][0] > 0
    assert r["share_first_wins_argmax_nontie"] == 1.0


def test_first_position_readout_symmetric_judge_has_zero_diff():
    from llm_preference_audit.bias import first_position_readout

    n = 400
    raw = np.tile([0.45, 0.45, 0.1], (n, 1))
    r = first_position_readout(raw, raw, np.arange(n) // 2)
    assert r["diff"] == pytest.approx(0.0)
    assert r["ci95"][0] <= 0 <= r["ci95"][1]
