"""Rung-6 input assembly, cost arithmetic and shared helpers. Torch-free: a fake tokenizer."""

import importlib.util
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_preference_audit.deberta_inputs import (
    OOF_COLUMNS,
    assert_no_text,
    oof_frame,
    prepare_rows,
    swap_example,
)
from llm_preference_audit.kernel_common import MANIFEST_SHA256, TRAIN_SHA256
from llm_preference_audit.lora_cost import (
    G0_TOTAL_CAP_USD,
    PRICE_PER_HOUR,
    est_cost_usd,
    project,
)
from llm_preference_audit.lora_inputs import (
    SEGMENTS,
    WINDOW_SLACK,
    build_lora_ids,
    field_window,
    make_template,
)

ROOT = Path(__file__).resolve().parents[1]
BOS = 2


def encode(s: str) -> list[int]:
    """Fake tokenizer: one token per whitespace word (stable ids >= 10), no special tokens."""
    return [10 + zlib.crc32(w.encode()) % 1000 for w in s.split()]


def count(s: str) -> int:
    return len(encode(s))


TEMPLATE = make_template(encode, BOS)


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_template_overhead_counts_bos_and_segments():
    assert TEMPLATE.bos == (BOS,)
    assert TEMPLATE.overhead == 1 + sum(count(s) for s in SEGMENTS)
    assert make_template(encode, None).overhead == TEMPLATE.overhead - 1


def test_assembly_layout():
    ids = build_lora_ids(TEMPLATE, [11, 12], [21], [31, 32, 33], limit=500)
    s0, s1, s2, s3 = TEMPLATE.segments
    assert ids == [BOS, *s0, 11, 12, *s1, 21, *s2, 31, 32, 33, *s3]
    assert ids[-len(s3) :] == list(s3)  # the question stays last: the pooled token


def test_window_respected_after_fit_and_cut():
    t = 300
    window = field_window(t, TEMPLATE)
    assert window == t - TEMPLATE.overhead - WINDOW_SLACK
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = rng.integers(1, 600, 3)
        p, a, b = (rng.integers(10, 99, k).tolist() for k in n)
        assert len(build_lora_ids(TEMPLATE, p, a, b, t)) <= t


def test_field_window_rejects_a_window_with_no_room():
    with pytest.raises(ValueError, match="no room"):
        field_window(TEMPLATE.overhead + WINDOW_SLACK, TEMPLATE)


def slots(ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Split an assembled sequence back into (prompt, a, b) field ids. Field ids are >= 2000 and
    template ids are below, so the segment boundaries are unambiguous."""
    s0, s1, s2, s3 = (list(s) for s in TEMPLATE.segments)
    head = len(TEMPLATE.bos) + len(s0)
    assert ids[: len(TEMPLATE.bos) + len(s0)] == [*TEMPLATE.bos, *s0]
    assert ids[-len(s3) :] == s3
    body = ids[head : -len(s3)]
    fields, cur, i = [], [], 0
    for seg in (s1, s2):
        while body[i : i + len(seg)] != seg:
            cur.append(body[i])
            i += 1
        fields.append(cur)
        cur, i = [], i + len(seg)
    fields.append(body[i:])
    return fields[0], fields[1], fields[2]


@pytest.mark.parametrize("sizes", [(5, 3, 4), (200, 40, 400), (400, 400, 400), (10, 700, 30)])
def test_swap_is_token_exact_even_when_hard_cut(sizes):
    rng = np.random.default_rng(sum(sizes))
    p, a, b = (rng.integers(2000, 3000, k).tolist() for k in sizes)
    for y in (0, 1, 2):
        sp, sa, sb, sy = swap_example(p, a, b, y)
        assert sy == (1, 0, 2)[y]
        orig = slots(build_lora_ids(TEMPLATE, p, a, b, limit=256))
        swapped = slots(build_lora_ids(TEMPLATE, sp, sa, sb, limit=256))
        # the swapped sequence is the original with the two (possibly cut) responses exchanged
        assert swapped == (orig[0], orig[2], orig[1])


def test_prepared_rows_fit_the_model_window():
    n = 12
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "prompt": [[words(300, "p")]] * n,
            "response_a": [[words(50 + 100 * i, "a")] for i in range(n)],
            "response_b": [[words(700, "b")]] * n,
            "label": pd.Categorical(["a"] * n, categories=["a", "b", "tie"]),
            "onehot_violation": [False] * n,
        }
    )
    manifest = pd.DataFrame(
        {"id": np.arange(n), "group": [f"g{i}" for i in range(n)], "fold": np.arange(n) % 5}
    ).assign(split="cv")
    t = 400
    rows = prepare_rows(df, manifest, count, field_window(t, TEMPLATE))
    for r in rows.itertuples():
        ids = build_lora_ids(
            TEMPLATE, encode(r.prompt_text), encode(r.a_text), encode(r.b_text), limit=t
        )
        assert len(ids) <= t
        assert r.n_prompt + r.n_a + r.n_b + TEMPLATE.overhead <= t or r.hard_cut


def test_oof_has_no_text_and_rung5_columns():
    rng = np.random.default_rng(1)
    P = rng.dirichlet([2, 2, 2], 6)
    frame = oof_frame(np.arange(6), P, P[:, [1, 0, 2]], [True] * 6, [False] * 6)
    assert list(frame.columns) == OOF_COLUMNS
    assert_no_text(frame)
    bad = frame.assign(a_text="x")
    with pytest.raises(ValueError, match="text"):
        assert_no_text(bad)


def test_shared_constants_are_the_rung5_values():
    assert TRAIN_SHA256.startswith("0692154a") and MANIFEST_SHA256.startswith("0a77276d")
    src = (ROOT / "kaggle/rung5_deberta/train.py").read_text()
    assert TRAIN_SHA256 not in src and MANIFEST_SHA256 not in src  # single source of truth


def test_cost_arithmetic():
    assert PRICE_PER_HOUR == 3.95
    assert est_cost_usd(3600) == pytest.approx(3.95)
    p = project(n_cv=51848, train_samples_per_s=10, infer_seqs_per_s=40, load_seconds=120)
    n_train = 51848 * 0.8 * 0.9
    assert p["fold_train_seconds"] == pytest.approx(n_train / 10)
    assert p["full_seconds"] == pytest.approx(120 + 51848 / 10)
    assert p["total_usd_with_overhead"] == pytest.approx((5 * p["fold_usd"] + p["full_usd"]) * 1.15)
    assert p["folds_1_to_4_plus_full_usd"] == pytest.approx(4 * p["fold_usd"] + p["full_usd"])
    assert p["meets_g0_total_cap"] == (p["total_usd_with_overhead"] <= G0_TOTAL_CAP_USD)
    slow = project(n_cv=51848, train_samples_per_s=1, infer_seqs_per_s=4, load_seconds=0)
    assert not slow["meets_g0_total_cap"]
    with pytest.raises(ValueError):
        project(n_cv=1, train_samples_per_s=0, infer_seqs_per_s=1, load_seconds=0)


def test_merge_oof_reads_rung6_wall_seconds():
    spec = importlib.util.spec_from_file_location(
        "merge_oof", ROOT / "kaggle/rung5_deberta/merge_oof.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.gpu_seconds({"wall_seconds": 100.0, "train_stats": {"train_seconds": 5}}) == 100.0
    assert mod.gpu_seconds({"train_stats": {"train_seconds": 5}, "predict_seconds": 2}) == 7.0
