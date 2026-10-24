"""Aggregate-only profile of the Chatbot Arena preference data.

Pure compute functions live here; file I/O is in scripts/make_profile.py. Nothing in
the output may contain prompt or response text.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from llm_preference_audit.data import TEXT_COLS, conversation_text
from llm_preference_audit.split import group_hash

TOKENS_PER_WORD = 1.33
TOKEN_THRESHOLDS = (4_000, 8_000, 16_000)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def class_prior_log_loss(labels: pd.Series) -> float:
    """Natural-log loss of predicting the empirical class shares for every row."""
    shares = labels.dropna().astype(str).value_counts(normalize=True)
    return float(-(shares * np.log(shares)).sum())


def repeat_agreement(df: pd.DataFrame) -> dict:
    """Label agreement among repeats of one prompt group with the same unordered model pair.

    Needs columns group, model_a, model_b, label. Outcomes are mapped to the winning
    model name (or "tie") so an A/B swap between repeats is compared correctly.
    """
    d = df.dropna(subset=["label"])
    winner = np.where(
        d["label"].astype(str) == "a",
        d["model_a"],
        np.where(d["label"].astype(str) == "b", d["model_b"], "tie"),
    )
    pair = [tuple(sorted(x)) for x in zip(d["model_a"], d["model_b"], strict=True)]
    work = pd.DataFrame({"group": d["group"].to_numpy(), "pair": pair, "outcome": winner})
    n_cells = n_pairs = n_agree = n_all_agree = 0
    for _, g in work.groupby(["group", "pair"], sort=False):
        if len(g) < 2:
            continue
        n_cells += 1
        outcomes = g["outcome"].tolist()
        n_all_agree += len(set(outcomes)) == 1
        for x, y in combinations(outcomes, 2):
            n_pairs += 1
            n_agree += x == y
    return {
        "n_groups": n_cells,
        "n_pairs": n_pairs,
        "agree_share": n_agree / n_pairs if n_pairs else math.nan,
        "groups_all_agree_share": n_all_agree / n_cells if n_cells else math.nan,
    }


def longer_wins(df: pd.DataFrame) -> dict:
    """Among non-ties with unequal word counts, how often the longer response won."""
    d = df[df["label"].astype(str).isin(["a", "b"]) & (df["words_a"] != df["words_b"])]
    longer_is_a = d["words_a"] > d["words_b"]
    won = int(((d["label"].astype(str) == "a") == longer_is_a).sum())
    n = len(d)
    lo, hi = wilson_interval(won, n)
    return {"n": n, "longer_won": won, "share": won / n if n else math.nan, "ci95": [lo, hi]}


def _dist(s: pd.Series) -> dict:
    return {
        "p50": float(s.quantile(0.5)),
        "p90": float(s.quantile(0.9)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["words_prompt"] = [len(conversation_text(t).split()) for t in df["prompt"]]
    out["words_a"] = [len(conversation_text(t).split()) for t in df["response_a"]]
    out["words_b"] = [len(conversation_text(t).split()) for t in df["response_b"]]
    out["n_turns"] = df["prompt"].map(len)
    out["est_tokens"] = (out.words_prompt + out.words_a + out.words_b) * TOKENS_PER_WORD
    out["group"] = [group_hash(t)[:16] for t in df["prompt"]]
    return out


def label_section(df: pd.DataFrame) -> dict:
    counts = df["label"].astype(str).value_counts().reindex(["a", "b", "tie"]).fillna(0)
    n = int(counts.sum())
    nonties = counts["a"] + counts["b"]
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "shares": {k: float(v / n) for k, v in counts.items()},
        "class_prior_log_loss": class_prior_log_loss(df["label"]),
        "position": {
            "n_nonties": int(nonties),
            "p_a_among_nonties": float(counts["a"] / nonties),
            "p_b_among_nonties": float(counts["b"] / nonties),
            "p_a_ci95": list(wilson_interval(int(counts["a"]), int(nonties))),
        },
    }


def model_section(df: pd.DataFrame) -> dict:
    appearances = pd.concat([df["model_a"], df["model_b"]]).value_counts()
    return {
        "n_distinct": len(appearances),
        "top10": [{"model": m, "appearances": int(c)} for m, c in appearances.head(10).items()],
    }


def duplicate_section(df: pd.DataFrame) -> dict:
    key_cols = ["model_a", "model_b", *TEXT_COLS, "winner_model_a", "winner_model_b", "winner_tie"]
    keys = df[key_cols].astype(str).agg("\x1f".join, axis=1)
    exact_dups = int(keys.duplicated().sum())
    sizes = df.groupby("group").size()
    full = repeat_agreement(df.assign(label=df["label"]))
    deduped = repeat_agreement(df[~keys.duplicated()])
    n_pairs = deduped["n_pairs"]
    return {
        "exact_duplicate_rows": exact_dups,
        "distinct_prompt_groups": len(sizes),
        "rows_in_groups_ge2": int(sizes[sizes >= 2].sum()),
        "groups_ge2": int((sizes >= 2).sum()),
        "repeat_same_model_pair": {"all_rows": full, "exact_duplicates_removed": deduped},
        "noise_ceiling": (
            f"n={n_pairs} label pairs from {deduped['n_groups']} repeated prompt/model-pair "
            "cells (exact duplicates removed). "
            + (
                "Enough for a rough overall agreement rate only: pairs within a cell are not "
                "independent (effective n is nearer the cell count), repeated prompts skew "
                "short and generic, and it is too small for per-model or per-length strata."
                if n_pairs >= 300
                else "Too small to support a noise-ceiling estimate."
            )
        ),
    }


def length_section(d: pd.DataFrame) -> dict:
    total = float(d.est_tokens.sum())
    return {
        "words": {
            "prompt": _dist(d.words_prompt),
            "response_a": _dist(d.words_a),
            "response_b": _dist(d.words_b),
        },
        "est_tokens": {
            "total": total,
            "per_row_mean": total / len(d),
            "per_row": _dist(d.est_tokens),
            "tokens_per_word_assumed": TOKENS_PER_WORD,
        },
        "share_over_est_tokens": {
            str(t): float((d.est_tokens > t).mean()) for t in TOKEN_THRESHOLDS
        },
    }


def split_section(d: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    m = manifest.merge(d[["id", "label"]], on="id", how="left")
    m["cell"] = np.where(m.split == "test", "test", "fold" + m.fold.astype(str))

    def one(g: pd.DataFrame) -> dict:
        shares = g["label"].astype(str).value_counts(normalize=True)
        return {
            "rows": len(g),
            "groups": int(g.group.nunique()),
            "label_shares": {k: float(shares.get(k, 0.0)) for k in ["a", "b", "tie"]},
        }

    return {cell: one(g) for cell, g in sorted(m.groupby("cell"))}


def compare_copies(hf: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Row counts, id set equality, label and raw-text mismatch counts on shared ids."""
    shared = sorted(set(hf["id"]) & set(other["id"]))
    a = hf.set_index("id").loc[shared]
    b = other.set_index("id").loc[shared]
    onehot = ["winner_model_a", "winner_model_b", "winner_tie"]
    return {
        "rows_hf": len(hf),
        "rows_other": len(other),
        "id_sets_equal": set(hf["id"]) == set(other["id"]),
        "shared_ids": len(shared),
        "label_mismatch_rows": int(
            (a[onehot].to_numpy() != b[onehot].to_numpy()).any(axis=1).sum()
        ),
        "text_mismatch_rows": int((a[list(TEXT_COLS)] != b[list(TEXT_COLS)]).any(axis=1).sum()),
    }


def build_profile(df: pd.DataFrame, manifest: pd.DataFrame, comparison: dict | None) -> dict:
    d = add_length_columns(df)
    multi = d.n_turns > 1
    return {
        "rows": len(d),
        "integrity": {
            "parse_failures": {c: int(d[f"{c}_parse_failed"].sum()) for c in TEXT_COLS},
            "rows_with_null_turns": {c: int((d[f"{c}_null_turns"] > 0).sum()) for c in TEXT_COLS},
            "null_turns_total": {c: int(d[f"{c}_null_turns"].sum()) for c in TEXT_COLS},
            "onehot_violations": int(d.onehot_violation.sum()),
        },
        "hf_vs_kaggle": comparison
        or {"status": "not run: Kaggle download refused (403), competition rules not accepted"},
        "labels": label_section(d),
        "turns": {
            "multi_turn_share": float(multi.mean()),
            "turn_count_distribution": {
                str(k): int(v) for k, v in d.n_turns.value_counts().sort_index().items()
            },
        },
        "length": length_section(d),
        "verbosity_longer_wins": longer_wins(d),
        "models": model_section(d),
        "duplicates": duplicate_section(d.assign(label=d["label"])),
        "splits": split_section(d, manifest),
    }
