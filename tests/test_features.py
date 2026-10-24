import numpy as np
import pandas as pd
import pytest

from llm_preference_audit.features import (
    LENGTH_ONLY,
    RESPONSE_FEATURES,
    build_features,
    feature_columns,
    response_features,
    swap_labels,
    swap_pair,
    swap_probs,
    symmetric_predict,
)


def test_counts_on_synthetic_text():
    text = (
        "# Title\nHello world, what is this?\n- one\n* two\n1. three\n2) four\n"
        "```\ncode\n```\n**bold** and **more** see https://a.example/x and http://b.example\n"
    )
    f = response_features(["hello world"], [text])
    assert f["n_headers"] == 1
    assert f["n_bullets"] == 4
    assert f["n_code_fences"] == 2
    assert f["n_bold"] == 2
    assert f["n_urls"] == 2
    assert f["n_questions"] == 1
    assert f["n_newlines"] == text.count("\n")
    assert f["n_chars"] == len(text)
    assert f["n_words"] == len(text.split())
    assert f["n_turns"] == 1


def test_multi_turn_and_null_turns_count_as_empty():
    f = response_features(["q"], ["one two", "", "three"])
    assert f["n_turns"] == 3
    assert f["n_words"] == 3
    empty = response_features(["q"], [""])
    assert empty["n_words"] == 0 and empty["n_chars"] == 0 and empty["prompt_overlap"] == 0.0
    none_like = response_features(["q"], [])
    assert none_like["n_turns"] == 0 and none_like["n_words"] == 0


def test_refusal_flag_and_count_case_insensitive():
    f = response_features(["q"], ["I'M SORRY, but I cannot help. As an AI, I can't."])
    assert f["refusal_flag"] == 1
    assert f["refusal_count"] == 4
    g = response_features(["q"], ["Sure, here you go."])
    assert g["refusal_flag"] == 0 and g["refusal_count"] == 0


def test_prompt_overlap_is_share_of_prompt_word_types():
    f = response_features(["Apple banana apple cherry"], ["I like APPLE and cherry pie"])
    assert f["prompt_overlap"] == pytest.approx(2 / 3)
    assert response_features([""], ["anything"])["prompt_overlap"] == 0.0


def synth(n=6):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "- item", "**b**", "?"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": 100 + i,
                "prompt": [" ".join(rng.choice(words, 5))],
                "response_a": [" ".join(rng.choice(words, rng.integers(0, 30)))],
                "response_b": [" ".join(rng.choice(words, rng.integers(0, 30))), ""],
                "model_a": f"m{i}",
                "model_b": f"n{i}",
            }
        )
    return pd.DataFrame(rows)


def test_pair_features_values_and_columns():
    X = build_features(synth())
    assert X.index.name == "id" and list(X.index) == list(range(100, 106))
    assert list(X.columns) == feature_columns("full")
    for k in RESPONSE_FEATURES:
        np.testing.assert_allclose(X[f"{k}_diff"], X[f"{k}_a"] - X[f"{k}_b"])
        np.testing.assert_allclose(X[f"{k}_absdiff"], (X[f"{k}_a"] - X[f"{k}_b"]).abs())
        np.testing.assert_allclose(
            X[f"{k}_logratio"], np.log1p(X[f"{k}_a"]) - np.log1p(X[f"{k}_b"])
        )
    assert set(LENGTH_ONLY) <= set(X.columns) and len(LENGTH_ONLY) == 5


def test_feature_matrix_has_no_model_identity():
    df = synth()
    X = build_features(df)
    assert not [c for c in X.columns if "model" in c.lower()]
    shuffled = df.copy()
    shuffled["model_a"] = shuffled["model_a"].sample(frac=1, random_state=3).to_numpy()
    shuffled["model_b"] = "zzz"
    pd.testing.assert_frame_equal(X, build_features(shuffled))
    pd.testing.assert_frame_equal(X, build_features(df.drop(columns=["model_a", "model_b"])))


def test_swap_twice_is_identity_and_flips_a_b():
    X = build_features(synth())
    S = swap_pair(X)
    pd.testing.assert_frame_equal(swap_pair(S), X)
    np.testing.assert_allclose(S["n_words_a"], X["n_words_b"])
    np.testing.assert_allclose(S["n_words_diff"], -X["n_words_diff"])
    np.testing.assert_allclose(S["n_words_logratio"], -X["n_words_logratio"])
    np.testing.assert_allclose(S["n_words_absdiff"], X["n_words_absdiff"])
    np.testing.assert_allclose(S["prompt_words"], X["prompt_words"])
    # swapping equals recomputing features with the response columns exchanged
    df = synth()
    df2 = df.rename(columns={"response_a": "response_b", "response_b": "response_a"})
    pd.testing.assert_frame_equal(S, build_features(df2))


def test_swap_works_on_the_length_only_subset():
    X = build_features(synth())[list(LENGTH_ONLY)]
    S = swap_pair(X)
    assert list(S.columns) == list(X.columns)
    np.testing.assert_allclose(S["n_words_a"], X["n_words_b"])
    np.testing.assert_allclose(S["n_words_diff"], -X["n_words_diff"])
    pd.testing.assert_frame_equal(swap_pair(S), X)


def test_swap_label_mapping():
    assert list(swap_labels(np.array([0, 1, 2, 0, 1]))) == [1, 0, 2, 1, 0]
    y = np.array([0, 1, 2, 1])
    np.testing.assert_array_equal(swap_labels(swap_labels(y)), y)
    p = np.array([[0.2, 0.5, 0.3]])
    np.testing.assert_allclose(swap_probs(p), [[0.5, 0.2, 0.3]])


def test_symmetric_average_is_exactly_symmetric():
    X = build_features(synth())

    def lopsided(frame):  # deliberately position-biased "model"
        a = 1.0 + frame["n_words_a"].to_numpy() + 3.0
        b = 1.0 + frame["n_words_b"].to_numpy()
        t = np.full(len(frame), 2.0)
        P = np.stack([a, b, t], axis=1)
        return P / P.sum(axis=1, keepdims=True)

    p = symmetric_predict(lopsided, X)
    p_swapped = symmetric_predict(lopsided, swap_pair(X))
    np.testing.assert_allclose(p_swapped, swap_probs(p), atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    a_eq_b = X.copy()
    for k in RESPONSE_FEATURES:
        a_eq_b[f"{k}_b"] = a_eq_b[f"{k}_a"]
        a_eq_b[f"{k}_diff"] = 0.0
        a_eq_b[f"{k}_absdiff"] = 0.0
        a_eq_b[f"{k}_logratio"] = 0.0
    q = symmetric_predict(lopsided, a_eq_b)
    np.testing.assert_allclose(q[:, 0], q[:, 1], atol=1e-12)
