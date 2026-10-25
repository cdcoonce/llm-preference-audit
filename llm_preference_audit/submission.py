"""Torch-free pieces of the rung-6 Kaggle submission notebook.

Everything here is testable in CI: input discovery under /kaggle/input, test.csv parsing (no
labels), the GPU load split, token-budget batching, unlabelled row encoding with the SAME fitting
code as training, the wall-clock pass planner, part merging, the optional matrix-scaling
calibrator, and the exact submission frame. No dataset text leaves this module: every frame that is
written passes `assert_no_text`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from llm_preference_audit.data import ONEHOT_COLS, parse_turns
from llm_preference_audit.deberta_inputs import assert_no_text, swap_probs_rows
from llm_preference_audit.kernel_common import sha256_file
from llm_preference_audit.truncate import MIDDLE_SHARE, fit_pair

SUBMISSION_COLUMNS = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
TEST_COLUMNS = ("id", "prompt", "response_a", "response_b")
PROB_CLIP = 1e-6
PART_COLUMNS = [
    "id",
    "raw_ab_a",
    "raw_ab_b",
    "raw_ab_tie",
    "raw_ba_a",
    "raw_ba_b",
    "raw_ba_tie",
]
NINE_HOURS_S = 9 * 3600
DEFAULT_LIMIT_S = 8 * 3600 + 15 * 60
SAFETY = 1.15  # throughput variance allowed when projecting a timing run to the full test set
WARMUP_BATCHES = 3
KAGGLE_ROOTS = (Path("/kaggle/input"),)


# ---------------------------------------------------------------- input discovery


def locate_input(
    name: str,
    roots: Sequence[Path | str] = KAGGLE_ROOTS,
    expect_sha: str | None = None,
    prefer: str | None = None,
) -> Path:
    """The file called `name` under `roots`. The mount path has moved between Kaggle images, so we
    glob. With `expect_sha` only a file with that sha256 qualifies (train.csv). If several remain,
    the ones whose path contains `prefer` win; if that does not make it unique we refuse."""
    found = sorted({p for r in roots for p in Path(r).rglob(name) if p.is_file()})
    if expect_sha is not None:
        found = [p for p in found if sha256_file(p) == expect_sha]
    if prefer is not None and len(found) > 1:
        preferred = [p for p in found if prefer in str(p)]
        found = preferred or found
    if not found:
        raise FileNotFoundError(f"no {name} under {[str(r) for r in roots]}")
    if len(found) > 1:
        raise FileNotFoundError(f"{len(found)} candidates for {name}: {[str(p) for p in found]}")
    return found[0]


def locate_model_dir(
    roots: Sequence[Path | str] = KAGGLE_ROOTS, model_type: str = "gemma2", hint: str | None = None
) -> Path:
    """The directory holding a config.json whose model_type is `model_type` (the attached Kaggle
    Model). `hint` (e.g. '9b-it') picks among several."""
    found = []
    for r in roots:
        for cfg in sorted(Path(r).rglob("config.json")):
            try:
                if json.loads(cfg.read_text()).get("model_type") == model_type:
                    found.append(cfg.parent)
            except (OSError, ValueError):
                continue
    if hint is not None and len(found) > 1:
        found = [p for p in found if hint in str(p)] or found
    if not found:
        raise FileNotFoundError(f"no {model_type} config.json under {[str(r) for r in roots]}")
    if len(found) > 1:
        raise FileNotFoundError(f"{len(found)} {model_type} models: {[str(p) for p in found]}")
    return found[0]


def load_test(path: str | Path) -> pd.DataFrame:
    """test.csv as (id, prompt, response_a, response_b) with the three text columns parsed to turn
    lists. Never looks for labels; any extra column is ignored and ids must be unique."""
    raw = pd.read_csv(path)
    missing = [c for c in TEST_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"input is missing columns {missing}")
    df = raw[list(TEST_COLUMNS)].copy()
    df["id"] = df["id"].astype(np.int64)
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids in the input file")
    for col in TEST_COLUMNS[1:]:
        df[col] = [parse_turns(v).turns for v in df[col]]
    return df


def class_prior(train_csv: str | Path) -> np.ndarray:
    """Class frequencies [a, b, tie] from the one-hot label columns only (no text is read)."""
    onehot = pd.read_csv(train_csv, usecols=list(ONEHOT_COLS))[list(ONEHOT_COLS)]
    counts = onehot.to_numpy(dtype=float)
    ok = counts.sum(axis=1) == 1
    if not ok.any():
        raise ValueError("no row with exactly one winner")
    p = counts[ok].mean(axis=0)
    return p / p.sum()


# ---------------------------------------------------------------- scheduling


def split_for_gpus(lengths: Sequence[float], n: int) -> list[np.ndarray]:
    """Row indices for each of `n` GPUs with balanced total length: greedy longest-first onto the
    least loaded GPU (ties to the lowest GPU). Deterministic; each share is sorted ascending."""
    if n < 1:
        raise ValueError("need at least one GPU")
    L = np.asarray(lengths, dtype=float)
    order = sorted(range(len(L)), key=lambda i: (-L[i], i))
    loads = [0.0] * n
    shares: list[list[int]] = [[] for _ in range(n)]
    for i in order:
        g = min(range(n), key=lambda k: (loads[k], k))
        shares[g].append(i)
        loads[g] += float(L[i])
    return [np.array(sorted(s), dtype=np.int64) for s in shares]


def token_budget_batches(
    lengths: Sequence[int], max_tokens: int, max_batch: int
) -> list[list[int]]:
    """Group positions 0..n-1 in the GIVEN order (callers pass lengths longest first) into batches
    whose padded size (rows x first, i.e. longest, length) stays within `max_tokens`; a single row
    longer than the budget still gets its own batch."""
    if max_tokens < 1 or max_batch < 1:
        raise ValueError("max_tokens and max_batch must be positive")
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i, n in enumerate(lengths):
        new_max = max(cur_max, int(n))
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * new_max > max_tokens):
            batches.append(cur)
            cur, new_max = [], int(n)
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def batch_cost(lengths: Sequence[int], batch: Sequence[int]) -> int:
    """Padded tokens of one batch, one order."""
    return len(batch) * max(int(lengths[i]) for i in batch)


def choose_passes(
    remaining_cost: float, rate: float | None, time_left: float, batches_done: int = 0
) -> int:
    """2 (both orders) while the projection fits the time left, else 1. `rate` is padded tokens
    per second per pass measured so far. Before WARMUP_BATCHES batches (no trustworthy rate) we
    keep both orders; with no time left at all we still return 1 and the caller decides to stop."""
    if time_left <= 0:
        return 1
    if rate is None or rate <= 0 or batches_done < WARMUP_BATCHES:
        return 2
    return 2 if 2 * remaining_cost / rate <= time_left else 1


class EncodedRows(NamedTuple):
    ids: np.ndarray
    p: list
    a: list
    b: list
    hard_cut: np.ndarray


def encode_unlabelled(
    df: pd.DataFrame,
    count: Callable[[str], int],
    tokenize: Callable[[list[str]], list[Sequence[int]]],
    window: int,
    middle_share: float = MIDDLE_SHARE,
) -> EncodedRows:
    """Training's fitting for rows without labels: `fit_pair` to `window` tokens, then the three
    fitted texts tokenized (no special tokens). `window` is lora_inputs.field_window(T, template)."""
    p_t, a_t, b_t, cut = [], [], [], []
    for r in df.itertuples(index=False):
        pf = fit_pair(r.prompt, r.response_a, r.response_b, window, count, middle_share)
        p_t.append(pf.prompt.truncated.text)
        a_t.append(pf.a.truncated.text)
        b_t.append(pf.b.truncated.text)
        cut.append(pf.hard_cut)

    def ids(texts: list[str]) -> list:
        return [np.asarray(x, dtype=np.int32) for x in tokenize(texts)]

    return EncodedRows(
        df["id"].to_numpy(dtype=np.int64), ids(p_t), ids(a_t), ids(b_t), np.asarray(cut, dtype=bool)
    )


# ---------------------------------------------------------------- merging and the submission


def _norm(P: np.ndarray) -> np.ndarray:
    P = np.clip(np.asarray(P, dtype=float), PROB_CLIP, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def check_ids(ids: Iterable[int], expected_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(list(ids), dtype=np.int64)
    expected = np.asarray(list(expected_ids), dtype=np.int64)
    if len(set(expected.tolist())) != len(expected):
        raise ValueError("expected ids are not unique")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate ids in the predictions")
    missing = set(expected.tolist()) - set(ids.tolist())
    extra = set(ids.tolist()) - set(expected.tolist())
    if missing or extra:
        raise ValueError(
            f"ids differ from the test ids: {len(missing)} missing, {len(extra)} extra"
        )
    return ids, expected


def submission_frame(
    ids: Iterable[int], P: np.ndarray, expected_ids: Iterable[int] | None = None
) -> pd.DataFrame:
    """The submission: id, winner_model_a, winner_model_b, winner_tie.

    `P` rows (columns a, b, tie) belong to `ids`. With `expected_ids` (the test ids in file order)
    the ids must be unique and equal that set exactly, and rows come out in that order whatever
    order the predictions arrived in. Probabilities are clipped to [1e-6, 1] and renormalised."""
    P = np.asarray(P, dtype=float)
    ids_arr = np.asarray(list(ids), dtype=np.int64)
    if P.shape != (len(ids_arr), 3):
        raise ValueError(f"P must have shape ({len(ids_arr)}, 3), got {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError("non-finite probabilities")
    if expected_ids is not None:
        ids_arr, expected = check_ids(ids_arr, expected_ids)
        pos = {int(i): k for k, i in enumerate(ids_arr)}
        P = P[[pos[int(i)] for i in expected]]
        ids_arr = expected
    elif len(set(ids_arr.tolist())) != len(ids_arr):
        raise ValueError("duplicate ids in the predictions")
    out = pd.DataFrame(_norm(P), columns=SUBMISSION_COLUMNS[1:])
    out.insert(0, "id", ids_arr)
    assert_no_text(out)
    return out


def prior_submission(ids: Iterable[int], prior: Sequence[float]) -> pd.DataFrame:
    """Every row gets the class prior: a valid fallback file."""
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (3,) or (prior < 0).any() or abs(prior.sum() - 1) > 1e-6:
        raise ValueError("prior must be three non-negative numbers summing to 1")
    ids = list(ids)
    return submission_frame(ids, np.tile(prior, (len(ids), 1)), ids)


class Merged(NamedTuple):
    ids: np.ndarray
    P: np.ndarray  # (n, 3) swap-averaged where both orders ran, NaN where no order ran
    n_orders: np.ndarray  # 0, 1 or 2 finite orders per row


def merge_parts(parts: Iterable[pd.DataFrame]) -> Merged:
    """Combine worker part frames. A row with both orders is swap-averaged; with one order that
    order is used as is; with none (never reached, or non-finite) it is NaN with n_orders 0."""
    frames = list(parts)
    for f in frames:
        assert_no_text(f)
        if list(f.columns) != PART_COLUMNS:
            raise ValueError(f"unexpected part columns: {list(f.columns)}")
    if not frames:
        return Merged(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64))
    m = pd.concat(frames, ignore_index=True)
    if m["id"].duplicated().any():
        raise ValueError("an id appears in more than one part")
    ab = m[PART_COLUMNS[1:4]].to_numpy(dtype=float)
    ba = swap_probs_rows(m[PART_COLUMNS[4:7]].to_numpy(dtype=float))
    ok_ab, ok_ba = np.isfinite(ab).all(axis=1), np.isfinite(ba).all(axis=1)
    P = np.full((len(m), 3), np.nan)
    both = ok_ab & ok_ba
    P[both] = (ab[both] + ba[both]) / 2
    P[ok_ab & ~ok_ba] = ab[ok_ab & ~ok_ba]
    P[~ok_ab & ok_ba] = ba[~ok_ab & ok_ba]
    return Merged(m["id"].to_numpy(dtype=np.int64), P, ok_ab.astype(int) + ok_ba.astype(int))


def fill_missing(merged: Merged, expected_ids: Sequence[int], prior: Sequence[float]):
    """Predictions for exactly `expected_ids`: rows with no result get the prior. Returns
    (ids, P, filled_mask) in the order of `expected_ids`."""
    expected = np.asarray(list(expected_ids), dtype=np.int64)
    extra = set(merged.ids.tolist()) - set(expected.tolist())
    if extra:
        raise ValueError(f"{len(extra)} predicted ids are not test ids")
    pos = {int(i): k for k, i in enumerate(merged.ids)}
    P = np.tile(np.asarray(prior, dtype=float), (len(expected), 1))
    filled = np.ones(len(expected), dtype=bool)
    for r, i in enumerate(expected):
        k = pos.get(int(i))
        if k is not None and np.isfinite(merged.P[k]).all():
            P[r], filled[r] = merged.P[k], False
    return expected, P, filled


# ---------------------------------------------------------------- calibrator


def apply_calibrator(P: np.ndarray, coef: dict | None) -> np.ndarray:
    """Apply a saved matrix-scaling calibrator (`calibrate.export_calibrator`) to raw
    probabilities; `None` returns them unchanged."""
    P = np.asarray(P, dtype=float)
    if coef is None:
        return P
    if coef.get("kind") != "matrix_scaling" or list(coef.get("classes", [])) != [0, 1, 2]:
        raise ValueError("unsupported calibrator")
    W, b = np.asarray(coef["coef"], dtype=float), np.asarray(coef["intercept"], dtype=float)
    if W.shape != (3, 3) or b.shape != (3,):
        raise ValueError("calibrator coefficients must be 3x3 with a length-3 intercept")
    floor = float(coef["prob_floor"])
    X = np.clip(P, floor, 1.0)
    X = np.log(X / X.sum(axis=1, keepdims=True))
    Z = X @ W.T + b
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- timing projection


def project_timing(
    n_rows: int,
    infer_s: float,
    encode_s: float,
    load_s: float,
    overhead_s: float,
    n_target: int = 25_000,
    limit_s: float = DEFAULT_LIMIT_S,
    passes: int = 2,
) -> dict:
    """Project a timing run (both orders, `n_rows` rows, the slowest worker's seconds) to
    `n_target` rows. Inference and encoding scale with rows; model load and overhead do not."""
    if n_rows < 1 or infer_s <= 0:
        raise ValueError("a timing projection needs rows and a positive inference time")
    scale = n_target / n_rows
    infer = infer_s * scale
    single = infer / max(passes, 1)
    total = infer + encode_s * scale + load_s + overhead_s
    total_single = single + encode_s * scale + load_s + overhead_s
    return {
        "n_timing_rows": n_rows,
        "n_target_rows": n_target,
        "rows_per_s_both_orders": n_rows / infer_s,
        "projected_infer_s": infer,
        "projected_encode_s": encode_s * scale,
        "projected_total_s": total,
        "projected_total_single_order_s": total_single,
        "safety_factor": SAFETY,
        "projected_total_with_safety_s": total * SAFETY,
        "limit_s": limit_s,
        "fits_budget": bool(total * SAFETY <= limit_s),
        "fits_9h_with_margin": bool(total * SAFETY <= NINE_HOURS_S),
        "single_order_fits_budget": bool(total_single * SAFETY <= limit_s),
    }


def percentile_summary(values: Sequence[float]) -> dict:
    v = np.asarray(list(values), dtype=float)
    if not len(v):
        return {}
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def human_seconds(s: float) -> str:
    h, rem = divmod(math.ceil(s), 3600)
    return f"{h}h{rem // 60:02d}m"
