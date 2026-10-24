import numpy as np
import pytest

from llm_preference_audit.judge.base import JudgeOutput
from llm_preference_audit.judge.harness import combine, make_items, refusal_rate, status_counts

PRIOR = np.array([0.4, 0.4, 0.2])


def ok(cid, a, b, t):
    w = ["A", "B", "tie"][int(np.argmax([a, b, t]))]
    return JudgeOutput(cid, "ok", a, b, t, w, 10, 5)


def bad(cid, status="refused"):
    return JudgeOutput(cid, status, None, None, None, None, 10, 0)


def test_make_items():
    items = make_items(["1", "2"], ["p1", "p2"], ["a1", "a2"], ["b1", "b2"])
    assert [i.custom_id for i in items] == ["1-ab", "1-ba", "2-ab", "2-ba"]
    assert (items[0].response_a, items[0].response_b) == ("a1", "b1")
    assert (items[1].response_a, items[1].response_b) == ("b1", "a1")
    assert items[3].prompt == "p2" and items[3].response_a == "b2"


def test_combine_swap_mapping():
    outs = [ok("1-ab", 0.6, 0.3, 0.1), ok("1-ba", 0.2, 0.7, 0.1)]
    r = combine(outs, ["1"], PRIOR)
    np.testing.assert_allclose(r["P_ab"], [[0.6, 0.3, 0.1]])
    np.testing.assert_allclose(r["P_ba"], [[0.7, 0.2, 0.1]])
    np.testing.assert_allclose(r["P"], [[0.65, 0.25, 0.1]])
    np.testing.assert_allclose(r["raw_ba"], [[0.2, 0.7, 0.1]])
    assert not r["fallback"][0] and r["status_ab"][0] == "ok"


def test_combine_refusal_falls_back_to_prior():
    outs = [ok("1-ab", 0.6, 0.3, 0.1), bad("1-ba")]
    r = combine(outs, ["1"], PRIOR)
    np.testing.assert_allclose(r["P"], [[0.5, 0.35, 0.15]])
    np.testing.assert_allclose(r["P_ba"], [PRIOR])
    assert r["fallback"][0] and r["status_ba"][0] == "refused"
    assert np.isnan(r["raw_ba"]).all() and not np.isnan(r["raw_ab"]).any()


def test_combine_both_bad_is_prior_and_asymmetric_prior_not_swapped():
    prior = np.array([0.5, 0.3, 0.2])
    r = combine([bad("1-ab"), bad("1-ba", "parse_failed")], ["1"], prior)
    np.testing.assert_allclose(r["P"], [prior])
    np.testing.assert_allclose(r["P_ba"], [prior])


def test_combine_order_follows_ids():
    outs = [
        ok("2-ba", 0.1, 0.8, 0.1),
        ok("1-ab", 0.5, 0.4, 0.1),
        ok("2-ab", 0.9, 0.05, 0.05),
        ok("1-ba", 0.4, 0.5, 0.1),
    ]
    r = combine(outs, ["1", "2"], PRIOR)
    assert r["P"].shape == (2, 3)
    np.testing.assert_allclose(r["P_ab"][:, 0], [0.5, 0.9])
    np.testing.assert_allclose(r["P_ba"][:, 0], [0.5, 0.8])


def test_combine_missing_or_duplicate_raises():
    with pytest.raises(ValueError, match="1-ba"):
        combine([ok("1-ab", 0.5, 0.4, 0.1)], ["1"], PRIOR)
    dup = [ok("1-ab", 0.5, 0.4, 0.1), ok("1-ab", 0.5, 0.4, 0.1), ok("1-ba", 0.5, 0.4, 0.1)]
    with pytest.raises(ValueError, match="1-ab"):
        combine(dup, ["1"], PRIOR)


def test_status_counts_and_refusal_rate():
    outs = [ok("a", 1, 0, 0), bad("b"), bad("c", "errored"), bad("d")]
    assert status_counts(outs) == {"ok": 1, "refused": 2, "errored": 1}
    assert refusal_rate(outs) == 0.5
    assert refusal_rate([]) == 0.0
