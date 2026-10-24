"""Hand-built features from raw (untruncated) turn lists.

Only prompt/response text is used. Model identity columns are never read.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from llm_preference_audit.data import conversation_text

REFUSAL_PHRASES = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "as an ai",
    "i am unable",
    "i'm unable",
)
RESPONSE_FEATURES = (
    "n_words",
    "n_chars",
    "n_turns",
    "n_newlines",
    "n_headers",
    "n_bullets",
    "n_code_fences",
    "n_bold",
    "n_urls",
    "n_questions",
    "refusal_flag",
    "refusal_count",
    "prompt_overlap",
)
PAIR_SUFFIXES = ("_a", "_b", "_diff", "_absdiff", "_logratio")
PROMPT_FEATURES = ("prompt_words", "prompt_turns")
LENGTH_ONLY = tuple(f"n_words{s}" for s in PAIR_SUFFIXES)

_HEADER = re.compile(r"^ {0,3}#{1,6}\s", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
_URL = re.compile(r"https?://\S+")
_WORD = re.compile(r"\w+")


def _word_types(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def response_features(prompt_turns: list[str], response_turns: list[str]) -> dict[str, float]:
    """Per-response features. Null turns are already "" (see data.parse_turns)."""
    text = conversation_text(response_turns)
    lower = text.casefold()
    refusals = sum(lower.count(p) for p in REFUSAL_PHRASES)
    prompt_types = _word_types(conversation_text(prompt_turns))
    overlap = len(prompt_types & _word_types(text)) / len(prompt_types) if prompt_types else 0.0
    return {
        "n_words": len(text.split()),
        "n_chars": len(text),
        "n_turns": len(response_turns),
        "n_newlines": text.count("\n"),
        "n_headers": len(_HEADER.findall(text)),
        "n_bullets": len(_BULLET.findall(text)),
        "n_code_fences": text.count("```"),
        "n_bold": len(_BOLD.findall(text)),
        "n_urls": len(_URL.findall(text)),
        "n_questions": text.count("?"),
        "refusal_flag": int(refusals > 0),
        "refusal_count": refusals,
        "prompt_overlap": overlap,
    }


def pair_features(fa: dict[str, float], fb: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in RESPONSE_FEATURES:
        a, b = fa[k], fb[k]
        out[f"{k}_a"] = a
        out[f"{k}_b"] = b
        out[f"{k}_diff"] = a - b
        out[f"{k}_absdiff"] = abs(a - b)
        out[f"{k}_logratio"] = float(np.log1p(a) - np.log1p(b))
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix indexed by id. Reads only id, prompt, response_a, response_b."""
    rows = []
    for rid, prompt, ra, rb in zip(
        df["id"], df["prompt"], df["response_a"], df["response_b"], strict=True
    ):
        feats = pair_features(response_features(prompt, ra), response_features(prompt, rb))
        feats["prompt_words"] = len(conversation_text(prompt).split())
        feats["prompt_turns"] = len(prompt)
        rows.append(feats)
    return pd.DataFrame(rows, index=pd.Index(df["id"].to_numpy(), name="id"))


def swap_pair(X: pd.DataFrame) -> pd.DataFrame:
    """Swap the roles of a and b: x_a<->x_b, diff and logratio flip sign, absdiff is kept."""
    out = X.copy()
    for k in RESPONSE_FEATURES:
        out[f"{k}_a"], out[f"{k}_b"] = X[f"{k}_b"], X[f"{k}_a"]
        out[f"{k}_diff"] = -X[f"{k}_diff"]
        out[f"{k}_logratio"] = -X[f"{k}_logratio"]
    return out


def swap_labels(y: np.ndarray) -> np.ndarray:
    """Class codes 0=a, 1=b, 2=tie: swapping a and b flips 0<->1 and keeps tie."""
    y = np.asarray(y)
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def swap_probs(p: np.ndarray) -> np.ndarray:
    """Map probabilities predicted for a swapped row back: p_a<-p_b', p_b<-p_a', tie<-tie'."""
    return np.asarray(p)[:, [1, 0, 2]]


def symmetric_predict(predict, X: pd.DataFrame) -> np.ndarray:
    """Average predict(X) with the mirrored prediction on the swapped row."""
    return (predict(X) + swap_probs(predict(swap_pair(X)))) / 2.0


def feature_columns(config: str) -> list[str]:
    if config == "length_only":
        return list(LENGTH_ONLY)
    cols = [f"{k}{s}" for k in RESPONSE_FEATURES for s in PAIR_SUFFIXES]
    return cols + list(PROMPT_FEATURES)
