"""Load the Chatbot Arena train.csv with robust parsing of the JSON turn lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd

TEXT_COLS = ("prompt", "response_a", "response_b")
ONEHOT_COLS = ("winner_model_a", "winner_model_b", "winner_tie")
LABELS = ["a", "b", "tie"]


class Parsed(NamedTuple):
    turns: list[str]
    failed: bool
    n_null: int


def parse_turns(value: object) -> Parsed:
    """Parse a JSON list of turn strings. Never raises.

    Null entries become "" and are counted. Malformed input yields the raw string
    as a single turn (or [] if there is none) with failed=True.
    """
    raw = value if isinstance(value, str) else ""
    fallback = [raw] if raw else []
    try:
        obj = json.loads(raw)
    except ValueError:
        return Parsed(fallback, True, 0)
    if not isinstance(obj, list):
        return Parsed(fallback, True, 0)
    n_null = sum(t is None for t in obj)
    turns = ["" if t is None else (t if isinstance(t, str) else str(t)) for t in obj]
    return Parsed(turns, False, n_null)


def conversation_text(turns: list[str]) -> str:
    return "\n".join(turns)


def label_from_onehot(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (categorical a/b/tie label, boolean one-hot violation mask).

    Rows without exactly one winner column set get a missing label.
    """
    onehot = df[list(ONEHOT_COLS)].astype(int)
    bad = onehot.sum(axis=1) != 1
    idx = onehot.to_numpy().argmax(axis=1)
    label = pd.Series(pd.Categorical.from_codes(idx, LABELS), index=df.index)
    label[bad] = pd.NA
    return label.astype(pd.CategoricalDtype(LABELS)), bad


def load_train(path: str | Path) -> pd.DataFrame:
    """Load train.csv. Text columns become lists of turns.

    Adds `label`, `onehot_violation`, and per text column `<col>_parse_failed`
    and `<col>_null_turns`.
    """
    df = pd.read_csv(path)
    for col in TEXT_COLS:
        parsed = [parse_turns(v) for v in df[col]]
        df[col] = [p.turns for p in parsed]
        df[f"{col}_parse_failed"] = [p.failed for p in parsed]
        df[f"{col}_null_turns"] = [p.n_null for p in parsed]
    df["label"], df["onehot_violation"] = label_from_onehot(df)
    return df
