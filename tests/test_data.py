import json

import pandas as pd

from llm_preference_audit.data import conversation_text, label_from_onehot, parse_turns


def test_parse_turns_ok():
    p = parse_turns(json.dumps(["a", "b"]))
    assert p.turns == ["a", "b"] and not p.failed and p.n_null == 0


def test_parse_turns_null_entries_recorded():
    p = parse_turns('["a", null, "c"]')
    assert p.turns == ["a", "", "c"] and p.n_null == 1 and not p.failed


def test_parse_turns_malformed_does_not_crash():
    for bad in ["{not json", "", None, float("nan"), '{"a": 1}', "42"]:
        p = parse_turns(bad)
        assert p.failed and isinstance(p.turns, list)


def test_conversation_text_joins_turns():
    assert conversation_text(["x", "y"]) == "x\ny"


def test_label_from_onehot_and_violations():
    df = pd.DataFrame(
        {
            "winner_model_a": [1, 0, 0, 1, 0],
            "winner_model_b": [0, 1, 0, 1, 0],
            "winner_tie": [0, 0, 1, 0, 0],
        }
    )
    label, bad = label_from_onehot(df)
    assert list(label[:3].astype(str)) == ["a", "b", "tie"]
    assert list(bad) == [False, False, False, True, True]
    assert label[3:].isna().all()
