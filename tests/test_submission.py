import json

import numpy as np
import pandas as pd
import pytest

from llm_preference_audit import submission as sub
from llm_preference_audit.calibrate import MatrixScaler, export_calibrator
from llm_preference_audit.kernel_common import sha256_file


def _csv(path, ids):
    pd.DataFrame(
        {
            "id": ids,
            "prompt": [json.dumps([f"p{i}"]) for i in ids],
            "response_a": [json.dumps([f"a{i}", None]) for i in ids],
            "response_b": ["not json" for _ in ids],
        }
    ).to_csv(path, index=False)


# ---------------------------------------------------------------- discovery


def test_locate_input_finds_wherever_the_mount_moved(tmp_path):
    deep = tmp_path / "competitions" / "llm-classification-finetuning"
    deep.mkdir(parents=True)
    _csv(deep / "test.csv", [1, 2])
    assert sub.locate_input("test.csv", [tmp_path]) == deep / "test.csv"


def test_locate_input_requires_the_known_sha_for_train(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "train.csv").write_text("x")
    (tmp_path / "b" / "train.csv").write_text("y")
    want = sha256_file(tmp_path / "b" / "train.csv")
    assert (
        sub.locate_input("train.csv", [tmp_path], expect_sha=want) == tmp_path / "b" / "train.csv"
    )
    with pytest.raises(FileNotFoundError):
        sub.locate_input("train.csv", [tmp_path], expect_sha="0" * 64)


def test_locate_input_refuses_ambiguity_unless_preferred(tmp_path):
    for d in ("other", "comp"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "test.csv").write_text("x")
    with pytest.raises(FileNotFoundError, match="2 candidates"):
        sub.locate_input("test.csv", [tmp_path])
    assert sub.locate_input("test.csv", [tmp_path], prefer="comp").parent.name == "comp"
    with pytest.raises(FileNotFoundError, match="no test.csv"):
        sub.locate_input("test.csv", [tmp_path / "nowhere"])


def test_locate_model_dir_reads_model_type(tmp_path):
    (tmp_path / "gemma").mkdir()
    (tmp_path / "gemma" / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
    (tmp_path / "bert").mkdir()
    (tmp_path / "bert" / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "config.json").write_text("{")
    assert sub.locate_model_dir([tmp_path]) == tmp_path / "gemma"
    with pytest.raises(FileNotFoundError):
        sub.locate_model_dir([tmp_path], model_type="llama")


def test_load_test_parses_turns_and_ignores_labels(tmp_path):
    p = tmp_path / "test.csv"
    _csv(p, [7, 3])
    df = sub.load_test(p)
    assert list(df.columns) == list(sub.TEST_COLUMNS)
    assert df["id"].tolist() == [7, 3]
    assert df["prompt"].iat[0] == ["p7"] and df["response_a"].iat[0] == ["a7", ""]
    assert df["response_b"].iat[0] == ["not json"]  # malformed JSON never raises


def test_load_test_rejects_missing_columns_and_duplicate_ids(tmp_path):
    pd.DataFrame({"id": [1], "prompt": ["[]"]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        sub.load_test(tmp_path / "x.csv")
    _csv(tmp_path / "d.csv", [1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        sub.load_test(tmp_path / "d.csv")


def test_class_prior_reads_only_label_columns(tmp_path):
    pd.DataFrame(
        {
            "winner_model_a": [1, 0, 0, 1, 1],
            "winner_model_b": [0, 1, 0, 0, 0],
            "winner_tie": [0, 0, 1, 0, 0],
        }
    ).to_csv(tmp_path / "t.csv", index=False)
    np.testing.assert_allclose(sub.class_prior(tmp_path / "t.csv"), [0.6, 0.2, 0.2])


# ---------------------------------------------------------------- scheduling


def test_split_for_gpus_partitions_and_balances():
    rng = np.random.default_rng(0)
    L = rng.integers(10, 2000, size=501)
    shares = sub.split_for_gpus(L, 2)
    assert sorted(np.concatenate(shares).tolist()) == list(range(501))
    loads = [L[s].sum() for s in shares]
    assert abs(loads[0] - loads[1]) <= L.max()  # greedy bound: gap at most one row
    assert abs(loads[0] - loads[1]) / L.sum() < 0.01


def test_split_for_gpus_is_deterministic_and_handles_edges():
    L = [5, 5, 5, 5, 1]
    a, b = sub.split_for_gpus(L, 2), sub.split_for_gpus(L, 2)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert [s.tolist() for s in sub.split_for_gpus([], 2)] == [[], []]
    assert sub.split_for_gpus([3, 4], 1)[0].tolist() == [0, 1]
    with pytest.raises(ValueError):
        sub.split_for_gpus([1], 0)


def test_token_budget_batches_respect_budget_and_order():
    L = [100, 90, 60, 50, 20, 10, 5]
    batches = sub.token_budget_batches(L, max_tokens=200, max_batch=3)
    assert [i for b in batches for i in b] == list(range(7))
    for b in batches:
        assert len(b) <= 3 and (len(b) == 1 or len(b) * L[b[0]] <= 200)
    assert batches[0] == [0, 1]
    assert sub.token_budget_batches([500], 100, 4) == [[0]]  # an over-budget row still runs


def test_choose_passes_switches_when_projection_overruns():
    assert sub.choose_passes(1000, rate=10, time_left=500, batches_done=10) == 2  # needs 200s
    assert sub.choose_passes(1000, rate=10, time_left=150, batches_done=10) == 1  # needs 200s
    assert sub.choose_passes(1000, rate=None, time_left=1, batches_done=0) == 2  # no rate yet
    assert sub.choose_passes(1000, rate=10, time_left=1, batches_done=1) == 2  # warm-up
    assert sub.choose_passes(1000, rate=10, time_left=0, batches_done=10) == 1


def test_encode_unlabelled_matches_the_training_fit_and_keeps_ids():
    df = pd.DataFrame(
        {
            "id": [5, 2],
            "prompt": [["hello there"], ["x " * 400]],
            "response_a": [["one two"], ["y " * 400]],
            "response_b": [["three"], ["z " * 400]],
        }
    )

    def count(s):
        return len(s.split())

    def tokenize(texts):
        return [[len(w) for w in t.split()] for t in texts]

    enc = sub.encode_unlabelled(df, count, tokenize, window=100)
    assert enc.ids.tolist() == [5, 2]
    assert [len(x) for x in (enc.p[0], enc.a[0], enc.b[0])] == [2, 2, 1]
    assert len(enc.a[1]) <= 100 and len(enc.b[1]) <= 100
    assert not enc.hard_cut[0]


# ---------------------------------------------------------------- submission frame


def test_submission_frame_columns_clip_and_sum():
    out = sub.submission_frame([3, 1], np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]), [3, 1])
    assert list(out.columns) == sub.SUBMISSION_COLUMNS
    assert out["id"].tolist() == [3, 1]
    vals = out.iloc[:, 1:].to_numpy()
    assert (vals >= 1e-6 * 0.99).all() and (vals <= 1).all()
    np.testing.assert_allclose(vals.sum(axis=1), 1.0, atol=1e-12)


def test_submission_frame_restores_expected_order_not_sorted_order():
    expected = [30, 10, 20]
    P = np.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25], [0.3, 0.3, 0.4]])  # for ids 10, 20, 30
    out = sub.submission_frame([10, 20, 30], P, expected)
    assert out["id"].tolist() == expected  # file order, not ascending
    np.testing.assert_allclose(out.iloc[0, 1:].to_numpy(), P[2])
    np.testing.assert_allclose(out.iloc[1, 1:].to_numpy(), P[0])


def test_submission_frame_refuses_missing_extra_duplicate_and_nonfinite():
    P = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError, match="1 missing"):
        sub.submission_frame([1, 2], P, [1, 2, 3])
    with pytest.raises(ValueError, match="1 extra"):
        sub.submission_frame([1, 9], P, [1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        sub.submission_frame([1, 1], P, [1, 2])
    with pytest.raises(ValueError, match="not unique"):
        sub.submission_frame([1, 2], P, [1, 1])
    with pytest.raises(ValueError, match="non-finite"):
        sub.submission_frame([1, 2], np.array([[np.nan, 0, 1], [0, 0, 1]]), [1, 2])
    with pytest.raises(ValueError, match="shape"):
        sub.submission_frame([1], P, [1])


def test_prior_submission_is_valid_and_constant():
    out = sub.prior_submission([4, 2, 9], [0.35, 0.34, 0.31])
    assert out["id"].tolist() == [4, 2, 9]
    np.testing.assert_allclose(out.iloc[:, 1:].to_numpy(), [[0.35, 0.34, 0.31]] * 3)
    with pytest.raises(ValueError):
        sub.prior_submission([1], [0.5, 0.5, 0.5])


@pytest.mark.parametrize("seed", range(25))
def test_property_rows_sum_to_one_and_order_matches_input(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    ids = rng.choice(10**9, size=n, replace=False).tolist()  # unsorted, unique
    P = rng.dirichlet(np.ones(3) * 0.2, size=n)  # includes near-degenerate rows
    perm = rng.permutation(n)
    out = sub.submission_frame([ids[k] for k in perm], P[perm], ids)
    assert out["id"].tolist() == ids
    vals = out.iloc[:, 1:].to_numpy()
    np.testing.assert_allclose(vals.sum(axis=1), 1.0, atol=1e-9)
    assert (vals >= 0).all() and np.isfinite(vals).all()
    np.testing.assert_allclose(
        vals, np.clip(P, 1e-6, 1) / np.clip(P, 1e-6, 1).sum(1, keepdims=True)
    )


# ---------------------------------------------------------------- merging


def _part(ids, ab=None, ba=None):
    n = len(ids)
    nan = np.full((n, 3), np.nan)
    ab = nan if ab is None else np.asarray(ab, dtype=float)
    ba = nan if ba is None else np.asarray(ba, dtype=float)
    return pd.DataFrame(np.hstack([ab, ba]), columns=sub.PART_COLUMNS[1:]).assign(
        id=np.asarray(ids, dtype=np.int64)
    )[sub.PART_COLUMNS]


def test_merge_parts_swap_averages_and_counts_orders():
    both = _part([1], ab=[[0.6, 0.3, 0.1]], ba=[[0.2, 0.5, 0.3]])  # ba is [p_b-first, ...]
    one = _part([2], ab=[[0.1, 0.2, 0.7]])
    only_ba = _part([3], ba=[[0.1, 0.8, 0.1]])
    none = _part([4])
    m = sub.merge_parts([both, one, only_ba, none])
    by = dict(zip(m.ids.tolist(), range(4), strict=True))
    np.testing.assert_allclose(m.P[by[1]], [(0.6 + 0.5) / 2, (0.3 + 0.2) / 2, (0.1 + 0.3) / 2])
    np.testing.assert_allclose(m.P[by[2]], [0.1, 0.2, 0.7])
    np.testing.assert_allclose(m.P[by[3]], [0.8, 0.1, 0.1])
    assert np.isnan(m.P[by[4]]).all()
    assert m.n_orders.tolist() == [2, 1, 1, 0]


def test_merge_parts_rejects_duplicates_and_text_columns():
    p = _part([1], ab=[[0.3, 0.3, 0.4]])
    with pytest.raises(ValueError, match="more than one part"):
        sub.merge_parts([p, p])
    with pytest.raises(ValueError, match="columns"):
        sub.merge_parts([p.assign(extra=1)])


def test_fill_missing_uses_the_prior_for_unreached_rows():
    m = sub.merge_parts([_part([5], ab=[[0.7, 0.2, 0.1]]), _part([6])])
    ids, P, filled = sub.fill_missing(m, [6, 5, 7], [0.4, 0.4, 0.2])
    assert ids.tolist() == [6, 5, 7] and filled.tolist() == [True, False, True]
    np.testing.assert_allclose(P[1], [0.7, 0.2, 0.1])
    np.testing.assert_allclose(P[0], [0.4, 0.4, 0.2])
    with pytest.raises(ValueError, match="not test ids"):
        sub.fill_missing(m, [6], [0.4, 0.4, 0.2])


# ---------------------------------------------------------------- calibrator


def test_apply_calibrator_reproduces_the_fitted_scaler_through_json():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(3), size=400)
    y = np.array([rng.choice(3, p=p**1.5 / (p**1.5).sum()) for p in P])
    scaler = MatrixScaler().fit(P, y)
    coef = json.loads(json.dumps(export_calibrator(scaler)))
    Q = rng.dirichlet(np.ones(3) * 0.5, size=60)
    np.testing.assert_allclose(sub.apply_calibrator(Q, coef), scaler.predict(Q), atol=1e-9)
    assert not np.allclose(sub.apply_calibrator(Q, coef), Q, atol=1e-3)  # it actually changes P


def test_apply_calibrator_none_is_identity_and_bad_shapes_raise():
    Q = np.array([[0.2, 0.3, 0.5]])
    np.testing.assert_array_equal(sub.apply_calibrator(Q, None), Q)
    good = {"kind": "matrix_scaling", "prob_floor": 1e-3, "classes": [0, 1, 2]}
    with pytest.raises(ValueError):
        sub.apply_calibrator(Q, good | {"coef": [[1, 0]], "intercept": [0, 0, 0]})
    with pytest.raises(ValueError, match="unsupported"):
        sub.apply_calibrator(Q, {"kind": "other"})


# ---------------------------------------------------------------- timing


def test_project_timing_scales_rows_not_fixed_costs():
    r = sub.project_timing(1000, infer_s=100, encode_s=10, load_s=300, overhead_s=60)
    assert r["projected_infer_s"] == pytest.approx(2500)
    assert r["projected_total_s"] == pytest.approx(2500 + 250 + 300 + 60)
    assert r["fits_budget"] and r["fits_9h_with_margin"]
    slow = sub.project_timing(1000, infer_s=1500, encode_s=10, load_s=300, overhead_s=60)
    assert not slow["fits_budget"] and not slow["fits_9h_with_margin"]
    assert slow["single_order_fits_budget"]
    with pytest.raises(ValueError):
        sub.project_timing(0, 1, 1, 1, 1)
