"""Pure (torch-free) pieces of the rung-5 DeBERTa pipeline: row prep, fold plans, OOF frames.

Text columns exist only in memory (prompt_text, a_text, b_text) and are never written; every
frame that leaves the process passes `assert_no_text`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from llm_preference_audit.cv_common import TestSplitLeak, assert_cv_only, inner_holdout_mask
from llm_preference_audit.data import LABELS
from llm_preference_audit.truncate import MIDDLE_SHARE, both_responses_whole, fit_pair

N_SPECIAL = 4  # [CLS] prompt [SEP] a [SEP] b [SEP]
OOF_COLUMNS = [
    "id",
    "p_a",
    "p_b",
    "p_tie",
    "raw_ab_a",
    "raw_ab_b",
    "raw_ab_tie",
    "raw_ba_a",
    "raw_ba_b",
    "raw_ba_tie",
    "both_untruncated",
    "hard_cut",
]
SUM_TOL = 1e-5
_TEXTY = re.compile(r"prompt|response|text", re.IGNORECASE)


def prepare_rows(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    count: Callable[[str], int],
    window: int,
    middle_share: float = MIDDLE_SHARE,
) -> pd.DataFrame:
    """Fit every labelled cv row to `window` tokens. `df` comes from `load_train` (turn lists).

    Returns id, y (0 a / 1 b / 2 tie), fold, group, the three fitted texts, hard_cut,
    both_untruncated, and the fitted token count per field (n_prompt, n_a, n_b).
    """
    m = manifest[manifest["split"] == "cv"][["id", "group", "fold"]]
    d = df[df["label"].notna() & ~df["onehot_violation"]]
    d = d.merge(m, on="id", how="inner").sort_values("id").reset_index(drop=True)
    rows = []
    for r in d.itertuples(index=False):
        pf = fit_pair(r.prompt, r.response_a, r.response_b, window, count, middle_share)
        rows.append(
            (
                int(r.id),
                LABELS.index(str(r.label)),
                int(r.fold),
                r.group,
                pf.prompt.truncated.text,
                pf.a.truncated.text,
                pf.b.truncated.text,
                pf.hard_cut,
                both_responses_whole(pf),
                pf.prompt.tokens,
                pf.a.tokens,
                pf.b.tokens,
            )
        )
    cols = [
        "id",
        "y",
        "fold",
        "group",
        "prompt_text",
        "a_text",
        "b_text",
        "hard_cut",
        "both_untruncated",
        "n_prompt",
        "n_a",
        "n_b",
    ]
    return pd.DataFrame(rows, columns=cols)


def fold_plan(
    rows: pd.DataFrame, fold: int, manifest: pd.DataFrame | None = None
) -> dict[str, np.ndarray]:
    """Ids to train on, to select epochs on (whole groups), and to predict, for outer fold `fold`.

    Fold `fold` is never in train or inner; train/inner/predict are group-disjoint.
    """
    if manifest is not None:
        assert_cv_only(rows["id"].to_numpy(), manifest)
    if "split" in rows.columns and (rows["split"] != "cv").any():
        raise TestSplitLeak("non-cv rows reached the fold plan")
    if rows["fold"].isna().any():
        raise TestSplitLeak("rows without a cv fold reached the fold plan")
    predict = rows[rows["fold"] == fold]
    train_all = rows[rows["fold"] != fold]
    if predict.empty or train_all.empty:
        raise ValueError(f"fold {fold} leaves an empty predict or train side")
    hold = inner_holdout_mask(train_all["group"].to_numpy())
    train, inner = train_all[~hold], train_all[hold]
    sets = {
        "train": set(train["group"]),
        "inner": set(inner["group"]),
        "predict": set(predict["group"]),
    }
    for a, b in (("train", "inner"), ("train", "predict"), ("inner", "predict")):
        if sets[a] & sets[b]:
            raise AssertionError(f"{a} and {b} share prompt groups")
    if set(predict["id"]) != set(rows.loc[rows["fold"] == fold, "id"]):  # pragma: no cover
        raise AssertionError("predict ids differ from the fold's rows")
    return {
        "train_ids": train["id"].to_numpy(),
        "inner_ids": inner["id"].to_numpy(),
        "predict_ids": predict["id"].to_numpy(),
    }


def swap_example(prompt_ids, a_ids, b_ids, y: int):
    """Swap the two responses; labels a<->b, tie stays."""
    return prompt_ids, b_ids, a_ids, (1, 0, 2)[int(y)]


def swap_probs_rows(P: np.ndarray) -> np.ndarray:
    """Map presented-order [p_a, p_b, p_tie] rows back/forth between orders."""
    return np.asarray(P)[:, [1, 0, 2]]


def oof_frame(
    ids: Sequence[int],
    raw_ab: np.ndarray,
    raw_ba: np.ndarray,
    both_untruncated: Sequence[bool],
    hard_cut: Sequence[bool],
) -> pd.DataFrame:
    """Per-row OOF frame. raw_* are presented-order probabilities (n, 3); final = swap average."""
    raw_ab, raw_ba = np.asarray(raw_ab, dtype=float), np.asarray(raw_ba, dtype=float)
    n = len(ids)
    for name, a in (("raw_ab", raw_ab), ("raw_ba", raw_ba)):
        if a.shape != (n, 3):
            raise ValueError(f"{name} must have shape ({n}, 3), got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} has non-finite probabilities")
        if (np.abs(a.sum(axis=1) - 1) > SUM_TOL).any():
            raise ValueError(f"{name} rows do not sum to 1")
    final = (raw_ab + swap_probs_rows(raw_ba)) / 2
    out = {"id": np.asarray(ids, dtype=np.int64)}
    for j, c in enumerate(("a", "b", "tie")):
        out[f"p_{c}"] = final[:, j]
    for order, raw in (("ab", raw_ab), ("ba", raw_ba)):
        for j, c in enumerate(("a", "b", "tie")):
            out[f"raw_{order}_{c}"] = raw[:, j]
    out["both_untruncated"] = np.asarray(both_untruncated, dtype=bool)
    out["hard_cut"] = np.asarray(hard_cut, dtype=bool)
    frame = pd.DataFrame(out)[OOF_COLUMNS]
    assert_no_text(frame)
    return frame


def assert_no_text(frame: pd.DataFrame) -> None:
    """Raise if the frame could carry dataset text (non-numeric dtype or a text-like name)."""
    for c in frame.columns:
        kind = frame[c].dtype.kind
        if kind not in "biuf" or _TEXTY.search(str(c)):
            raise ValueError(f"column {c!r} (dtype {frame[c].dtype}) may carry dataset text")


def merge_oof(frames: Iterable[pd.DataFrame], cv_ids: Iterable[int]) -> pd.DataFrame:
    """Concatenate per-fold OOF frames; every cv id must appear exactly once, nothing else."""
    frames = list(frames)
    for f in frames:
        assert_no_text(f)
        if list(f.columns) != OOF_COLUMNS:
            raise ValueError(f"unexpected OOF columns: {list(f.columns)}")
    merged = pd.concat(frames, ignore_index=True)
    dup = int(merged["id"].duplicated().sum())
    if dup:
        raise ValueError(f"{dup} ids appear in more than one fold")
    want, got = set(map(int, cv_ids)), set(merged["id"].astype(int))
    if got - want:
        raise ValueError(f"{len(got - want)} ids are not cv ids")
    if want - got:
        raise ValueError(f"{len(want - got)} cv ids are missing")
    return merged.sort_values("id").reset_index(drop=True)


def water_level_cut(lengths: Sequence[int], budget: int) -> list[int]:
    """Final (prompt, a, b) token counts after a symmetric water-level cut to `budget` tokens.

    Every field is capped at a common level h (the largest with sum(min(L, h)) <= budget). Leftover
    tokens go one each to capped fields, never favouring a or b by position: the prompt first, then
    a and b together or not at all (so the result may end one token under the budget).
    """
    L = [int(x) for x in lengths]
    if sum(L) <= budget:
        return L
    if budget < 0:
        raise ValueError("negative token budget")
    lo, hi = 0, max(L)
    while lo < hi:  # largest h with sum(min(L, h)) <= budget
        mid = (lo + hi + 1) // 2
        if sum(min(x, mid) for x in L) <= budget:
            lo = mid
        else:
            hi = mid - 1
    h = lo
    out = [min(x, h) for x in L]
    rest = budget - sum(out)
    if rest > 0 and L[0] > h:
        out[0] += 1
        rest -= 1
    a_cap, b_cap = L[1] > h, L[2] > h
    if a_cap and b_cap:
        if rest >= 2:
            out[1] += 1
            out[2] += 1
    elif a_cap and rest >= 1:
        out[1] += 1
    elif b_cap and rest >= 1:
        out[2] += 1
    return out


def build_input_ids(prompt_ids, a_ids, b_ids, cls_id: int, sep_id: int, limit: int) -> list[int]:
    """[CLS] prompt [SEP] a [SEP] b [SEP]; over `limit`, a symmetric water-level hard cut from ends."""
    fields = [list(prompt_ids), list(a_ids), list(b_ids)]
    budget = limit - N_SPECIAL
    if budget < 0:
        raise ValueError(f"limit {limit} is smaller than the special tokens")
    keep = water_level_cut([len(f) for f in fields], budget)
    p, a, b = (f[:k] for f, k in zip(fields, keep, strict=True))
    return [cls_id, *p, sep_id, *a, sep_id, *b, sep_id]


def length_grouped_batches(
    lengths: Sequence[int], batch_size: int, seed: int, pool_batches: int = 50
) -> list[np.ndarray]:
    """Shuffle, sort within pools of `pool_batches` batches by length, shuffle batch order."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    order = rng.permutation(len(lengths))
    pool = batch_size * pool_batches
    batches: list[np.ndarray] = []
    for s in range(0, len(order), pool):
        chunk = order[s : s + pool]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    perm = rng.permutation(len(batches))
    return [batches[i] for i in perm]


CONFIG_KEYS = ("git_sha", "t", "middle_share", "model", "epochs")


def validate_fold_parts(
    parts: Sequence[tuple[int, pd.DataFrame, dict]], manifest_cv: pd.DataFrame
) -> pd.DataFrame:
    """Merge per-fold OOF parts after checking each against its log and the split manifest.

    `parts` is (fold, oof frame, log dict). Requires: no logged error, ids of fold k exactly the
    manifest's fold-k cv ids, one shared value of CONFIG_KEYS across all logs, folds 0-4 once each.
    """
    folds_in_manifest = sorted(int(f) for f in manifest_cv["fold"].unique())
    seen: list[int] = []
    for fold, frame, log in parts:
        if log.get("error"):
            raise ValueError(f"fold {fold} log records an error: {log['error']}")
        if int(log.get("fold", fold)) != int(fold):
            raise ValueError(f"fold {fold} file has a log for fold {log.get('fold')}")
        seen.append(int(fold))
        want = set(manifest_cv.loc[manifest_cv["fold"] == fold, "id"].astype(int))
        got = frame["id"].astype(int)
        if got.duplicated().any():
            raise ValueError(f"fold {fold} has duplicate ids")
        if set(got) != want:
            raise ValueError(
                f"fold {fold} ids differ from the manifest's fold {fold}: "
                f"{len(set(got) - want)} foreign, {len(want - set(got))} missing"
            )
    dup = sorted({f for f in seen if seen.count(f) > 1})
    if dup:
        raise ValueError(f"fold(s) {dup} supplied more than once")
    if sorted(seen) != folds_in_manifest:
        raise ValueError(f"missing fold(s) {sorted(set(folds_in_manifest) - set(seen))}")
    for key in CONFIG_KEYS:
        vals = {json_key(log.get(key)) for _, _, log in parts}
        if len(vals) != 1:
            raise ValueError(f"config mismatch across folds on {key!r}: {sorted(vals)}")
    return merge_oof([f for _, f, _ in parts], manifest_cv["id"])


def json_key(v) -> str:
    return repr(v)
