import numpy as np
import pandas as pd
import pytest

from llm_preference_audit.deberta_inputs import (
    OOF_COLUMNS,
    assert_no_text,
    build_input_ids,
    fold_plan,
    length_grouped_batches,
    merge_oof,
    oof_frame,
    prepare_rows,
    swap_example,
    validate_fold_parts,
    water_level_cut,
)
from llm_preference_audit.train_lgbm import TestSplitLeak


def count(s: str) -> int:
    return len(s.split())


def _rows(n: int = 600) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "y": np.arange(n) % 3,
            "fold": np.arange(n) // 2 % 5,
            "group": [f"g{i // 2}" for i in range(n)],  # pairs share a group, same fold
            "hard_cut": False,
            "both_untruncated": True,
        }
    )


def _df(prompts, a, b) -> pd.DataFrame:
    n = len(prompts)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "prompt": prompts,
            "response_a": a,
            "response_b": b,
            "label": pd.Categorical(["a", "b", "tie"][:1] * n, categories=["a", "b", "tie"]),
            "onehot_violation": [False] * n,
        }
    )


def _manifest(n: int, split: str = "cv") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "group": [f"g{i}" for i in range(n)],
            "split": split,
            "fold": pd.array([i % 5 for i in range(n)], dtype="Int64"),
        }
    )


def test_fold_plan_disjoint_and_excludes_fold():
    rows = _rows()
    for k in range(5):
        plan = fold_plan(rows, k)
        tr, inn, pr = (set(plan[x]) for x in ("train_ids", "inner_ids", "predict_ids"))
        assert not (tr & inn) and not (tr & pr) and not (inn & pr)
        folds = rows.set_index("id")["fold"]
        assert all(folds[i] != k for i in tr | inn)
        assert pr == set(rows.id[rows.fold == k])
        assert tr | inn | pr == set(rows.id)
        groups = rows.set_index("id")["group"]
        g = [{groups[i] for i in s} for s in (tr, inn, pr)]
        assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
        assert len(inn) > 0


def test_fold_plan_rejects_non_cv():
    rows = _rows(100)
    with pytest.raises(TestSplitLeak):
        fold_plan(rows, 0, manifest=_manifest(100, "test"))
    with pytest.raises(TestSplitLeak):
        fold_plan(rows.assign(split="test"), 0)
    bad = rows.copy()
    bad["fold"] = bad["fold"].astype(float)
    bad.loc[3, "fold"] = np.nan
    with pytest.raises(TestSplitLeak):
        fold_plan(bad, 0)
    with pytest.raises(TestSplitLeak):  # id absent from the manifest
        fold_plan(rows, 0, manifest=_manifest(50))


def test_fold_plan_deterministic():
    a, b = fold_plan(_rows(), 2), fold_plan(_rows(), 2)
    for k in a:
        assert (a[k] == b[k]).all()


def test_oof_frame_mapping_and_sums():
    ab = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    ba = np.array([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]])  # presented order: col a = shown first
    f = oof_frame([10, 11], ab, ba, [True, False], [False, True])
    assert list(f.columns) == OOF_COLUMNS
    # ba mapped back to original order is [p_b, p_a, p_tie] = [0.7, 0.1, 0.2], [0.1, 0.5, 0.4]
    np.testing.assert_allclose(f[["p_a", "p_b", "p_tie"]].to_numpy()[0], [0.65, 0.2, 0.15])
    np.testing.assert_allclose(f[["p_a", "p_b", "p_tie"]].to_numpy()[1], [0.15, 0.35, 0.5])
    np.testing.assert_allclose(f[["p_a", "p_b", "p_tie"]].sum(axis=1), 1.0)
    assert f.loc[1, "raw_ba_a"] == 0.5 and f.loc[0, "raw_ab_tie"] == 0.1
    assert f["both_untruncated"].tolist() == [True, False] and f["hard_cut"].tolist() == [
        False,
        True,
    ]


def test_oof_frame_rejects_bad_probs():
    ok = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError):
        oof_frame([1], ok * 2, ok, [True], [False])
    with pytest.raises(ValueError):
        oof_frame([1], np.array([[np.nan, 0.5, 0.5]]), ok, [True], [False])


def test_assert_no_text_catches_text():
    f = oof_frame([1], [[0.5, 0.3, 0.2]], [[0.5, 0.3, 0.2]], [True], [False])
    assert_no_text(f)
    with pytest.raises(ValueError):
        assert_no_text(f.assign(note=["hello"]))
    with pytest.raises(ValueError):
        assert_no_text(f.assign(prompt_text=[1]))
    with pytest.raises(ValueError):
        assert_no_text(f.assign(c=pd.Series(["x"], dtype=object)))


def test_swap_example():
    p, a, b = [1], [2, 2], [3]
    assert swap_example(p, a, b, 0) == (p, b, a, 1)
    assert swap_example(p, a, b, 1) == (p, b, a, 0)
    assert swap_example(p, a, b, 2) == (p, b, a, 2)


def test_prepare_rows_flags_and_determinism():
    long_a = " ".join(f"w{i}." for i in range(400))
    df = _df([["hi there"], ["hi there"]], [["short"], [long_a]], [["short too"], [long_a]])
    m = _manifest(2)
    r1 = prepare_rows(df, m, count, window=60)
    r2 = prepare_rows(df, m, count, window=60)
    pd.testing.assert_frame_equal(r1, r2)
    assert r1["both_untruncated"].tolist() == [True, False]
    assert r1["hard_cut"].tolist() == [False, False]
    assert (r1["n_prompt"] + r1["n_a"] + r1["n_b"] <= 60).all()
    assert r1["y"].tolist() == [0, 0] and r1["fold"].tolist() == [0, 1]
    # a window too small for the 25% prompt floor plus markers forces a hard cut
    tiny = prepare_rows(df, m, count, window=3)
    assert tiny["hard_cut"].tolist()[1]


def test_prepare_rows_cv_only_and_labelled():
    df = _df([["p"]] * 4, [["a"]] * 4, [["b"]] * 4)
    df.loc[1, "onehot_violation"] = True
    m = _manifest(4)
    m.loc[2, "split"] = "test"
    assert prepare_rows(df, m, count, 50)["id"].tolist() == [0, 3]


def test_build_input_ids_and_hard_cut():
    ids = build_input_ids([1, 2], [3], [4, 5], cls_id=100, sep_id=101, limit=50)
    assert ids == [100, 1, 2, 101, 3, 101, 4, 5, 101]
    cut = build_input_ids([1, 2], [3] * 10, [4] * 6, 100, 101, limit=12)
    assert len(cut) == 12 and cut[0] == 100 and cut[-1] == 101
    assert cut.count(3) < 10 and cut.count(1) == 1 and cut.count(4) <= 6
    with pytest.raises(ValueError):
        build_input_ids([], [], [], 100, 101, limit=3)


def test_length_grouped_batches():
    lengths = np.random.default_rng(0).integers(5, 500, size=1000)
    b1 = length_grouped_batches(lengths, 4, seed=7)
    b2 = length_grouped_batches(lengths, 4, seed=7)
    assert all((x == y).all() for x, y in zip(b1, b2, strict=True))
    assert sorted(np.concatenate(b1).tolist()) == list(range(1000))
    assert all(len(b) == 4 for b in b1)
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in b1])
    assert spread < 100  # far below the ~490 of random batches
    b3 = length_grouped_batches(lengths, 4, seed=8)
    assert not all((x == y).all() for x, y in zip(b1, b3, strict=True))


def test_merge_oof_coverage():
    def fr(ids):
        n = len(ids)
        p = np.tile([0.5, 0.3, 0.2], (n, 1))
        return oof_frame(ids, p, p, [True] * n, [False] * n)

    m = merge_oof([fr([2, 0]), fr([1])], [0, 1, 2])
    assert m["id"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="more than one"):
        merge_oof([fr([0, 1]), fr([1, 2])], [0, 1, 2])
    with pytest.raises(ValueError, match="missing"):
        merge_oof([fr([0, 1])], [0, 1, 2])
    with pytest.raises(ValueError, match="not cv"):
        merge_oof([fr([0, 1, 2, 9])], [0, 1, 2])


def _cut(p, a, b, limit):
    ids = build_input_ids([1] * p, [2] * a, [3] * b, 100, 101, limit)
    return ids.count(1), ids.count(2), ids.count(3), len(ids)


def test_hard_cut_example_is_balanced():
    p, a, b, n = _cut(100, 900, 900, 1024)
    assert (p, a, b) == (100, 460, 460) and n <= 1024


def test_hard_cut_symmetric_bounded_and_nonempty():
    for p in (0, 1, 50, 100, 400):
        for a in (0, 1, 7, 300, 900, 1500):
            for b in (0, 1, 8, 300, 900, 1500):
                for limit in (4, 5, 6, 64, 513, 1024):
                    if p + a + b + 4 > limit and limit < 4:
                        continue
                    r1 = _cut(p, a, b, limit)
                    r2 = _cut(p, b, a, limit)
                    assert (r1[0], r1[1], r1[2]) == (r2[0], r2[2], r2[1]), (p, a, b, limit)
                    assert r1[3] <= limit
                    kept = (r1[0], r1[1], r1[2])
                    if p + a + b + 4 > limit and limit - 4 >= 3:
                        # nothing emptied unless it started empty
                        assert all(k > 0 or o == 0 for k, o in zip(kept, (p, a, b), strict=True))
                    if p + a + b + 4 <= limit:
                        assert kept == (p, a, b)
                    else:
                        assert r1[3] >= limit - 1  # at most one token under the limit


def test_water_level_cut_ties_never_favour_a():
    assert water_level_cut([0, 5, 5], 5) == [0, 2, 2]
    assert water_level_cut([0, 5, 5], 7) == [0, 3, 3]
    assert water_level_cut([3, 9, 9], 10) == [3, 3, 3]
    assert water_level_cut([2, 9, 3], 8) == [2, 3, 3]
    with pytest.raises(ValueError):
        water_level_cut([1], -1)


def _fold_manifest():
    rows = [(i, i % 5) for i in range(10)]
    return pd.DataFrame(rows, columns=["id", "fold"])


def _part(fold, ids=None, **log):
    m = _fold_manifest()
    ids = list(m.loc[m["fold"] == fold, "id"]) if ids is None else ids
    n = len(ids)
    p = np.tile([0.5, 0.3, 0.2], (n, 1))
    cfg = {"fold": fold, "git_sha": "s", "t": 1024, "middle_share": 0.2, "model": "m", "epochs": 2}
    return fold, oof_frame(ids, p, p, [True] * n, [False] * n), cfg | log


def test_validate_fold_parts_happy_path():
    out = validate_fold_parts([_part(k) for k in (3, 0, 1, 2, 4)], _fold_manifest())
    assert out["id"].tolist() == list(range(10))


def test_validate_fold_parts_rejections():
    m = _fold_manifest()
    ok = [_part(k) for k in range(5)]
    wrong = [_part(0, ids=[1, 6])] + ok[1:]  # fold 1's ids in fold 0's file
    with pytest.raises(ValueError, match="differ from the manifest"):
        validate_fold_parts(wrong, m)
    with pytest.raises(ValueError, match="missing fold"):
        validate_fold_parts(ok[:4], m)
    with pytest.raises(ValueError, match="more than once"):
        validate_fold_parts([*ok, _part(2)], m)
    with pytest.raises(ValueError, match="middle_share"):
        validate_fold_parts([*ok[:4], _part(4, middle_share=0.0)], m)
    with pytest.raises(ValueError, match="git_sha"):
        validate_fold_parts([*ok[:4], _part(4, git_sha="x")], m)
    with pytest.raises(ValueError, match="error"):
        validate_fold_parts([*ok[:4], _part(4, error="OOM")], m)
