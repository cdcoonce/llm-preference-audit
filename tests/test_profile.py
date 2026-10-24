import math

import pandas as pd
import pytest

from llm_preference_audit.profile import (
    class_prior_log_loss,
    compare_copies,
    longer_wins,
    repeat_agreement,
    wilson_interval,
)


def test_wilson_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


def test_class_prior_log_loss_is_entropy():
    labels = pd.Series(["a", "b", "tie", "tie"])
    expected = -(0.25 * math.log(0.25) * 2 + 0.5 * math.log(0.5))
    assert class_prior_log_loss(labels) == pytest.approx(expected)
    assert class_prior_log_loss(pd.Series(["a"] * 5)) == pytest.approx(0.0)


def frame(rows):
    return pd.DataFrame(rows, columns=["group", "model_a", "model_b", "label"])


def test_repeat_agreement_maps_swapped_repeats():
    # X beats Y once as A and once as B: the same outcome, must agree.
    df = frame([("g", "X", "Y", "a"), ("g", "Y", "X", "b")])
    r = repeat_agreement(df)
    assert r["n_groups"] == 1 and r["n_pairs"] == 1 and r["agree_share"] == 1.0


def test_repeat_agreement_swapped_same_label_disagrees():
    # "a won" twice but sides swapped: X won then Y won.
    df = frame([("g", "X", "Y", "a"), ("g", "Y", "X", "a")])
    assert repeat_agreement(df)["agree_share"] == 0.0


def test_repeat_agreement_ties_and_ignored_groups():
    df = frame(
        [
            ("g1", "X", "Y", "tie"),
            ("g1", "Y", "X", "tie"),
            ("g2", "X", "Y", "a"),  # singleton: ignored
            ("g3", "X", "Y", "a"),  # same group, different model pair: not a repeat
            ("g3", "X", "Z", "a"),
        ]
    )
    r = repeat_agreement(df)
    assert r["n_groups"] == 1 and r["n_pairs"] == 1 and r["agree_share"] == 1.0


def test_repeat_agreement_counts_all_pairs_in_triple():
    df = frame([("g", "X", "Y", "a"), ("g", "X", "Y", "a"), ("g", "Y", "X", "a")])
    r = repeat_agreement(df)
    assert r["n_pairs"] == 3 and r["agree_share"] == pytest.approx(1 / 3)


def test_longer_wins_excludes_ties_and_equal_length():
    df = pd.DataFrame(
        {
            "label": ["a", "b", "tie", "a", "b"],
            "words_a": [10, 10, 10, 5, 7],
            "words_b": [5, 5, 5, 5, 3],
        }
    )
    r = longer_wins(df)
    # a longer & a won (counts); a longer & b won (x2); tie and equal length skipped
    assert r["n"] == 3 and r["longer_won"] == 1


def copies(rows):
    cols = [
        "id",
        "prompt",
        "response_a",
        "response_b",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_compare_copies_counts_label_and_text_mismatches():
    base = [
        (1, ["p1"], ["ra"], ["rb"], 1, 0, 0),
        (2, ["p2"], ["ra"], ["rb"], 0, 1, 0),
        (3, ["p3"], ["ra"], ["rb"], 0, 0, 1),
        (5, ["p5"], ["ra"], ["rb"], 0, 1, 0),
    ]
    other = [
        (1, ["p1"], ["ra"], ["rb"], 1, 0, 0),
        (2, ["p2"], ["ra"], ["rb"], 1, 0, 0),
        (3, ["p3"], ["ra", "extra"], ["rb"], 0, 0, 1),
        (4, ["p4"], ["ra"], ["rb"], 1, 0, 0),
        (5, ["p5"], ["ra"], ["rb"], 0, 0, 1),  # b -> tie: winner_model_a unchanged
    ]
    cmp = compare_copies(copies(base), copies(other))
    assert cmp["rows_hf"] == 4 and cmp["rows_other"] == 5
    assert cmp["id_sets_equal"] is False and cmp["shared_ids"] == 4
    assert cmp["label_mismatch_rows"] == 2
    assert cmp["text_mismatch_rows"] == 1


def test_compare_copies_identical_is_clean():
    rows = [(1, ["p1"], ["ra"], ["rb"], 1, 0, 0), (2, ["p2"], ["ra"], ["rb"], 0, 0, 1)]
    cmp = compare_copies(copies(rows), copies(rows))
    assert cmp["id_sets_equal"] is True
    assert cmp["label_mismatch_rows"] == 0 and cmp["text_mismatch_rows"] == 0
