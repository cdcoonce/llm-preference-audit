"""Tests for the shared judge/model truncation (synthetic text only)."""

from __future__ import annotations

import math
import re
from itertools import pairwise

from llm_preference_audit.data import conversation_text
from llm_preference_audit.profile import TOKENS_PER_WORD
from llm_preference_audit.truncate import (
    HEAD_SHARE,
    MARKER_WORDS,
    TAIL_SHARE,
    Truncated,
    omission_marker,
    truncate_text,
    truncate_turns,
)

MARKER_RE = re.compile(r"\[\.\.\. (\d+) words omitted \.\.\.\]")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n")


def budget_for(word_budget: int) -> int:
    """Smallest token budget whose word budget is exactly `word_budget`."""
    b = math.ceil(word_budget * TOKENS_PER_WORD)
    while int(b / TOKENS_PER_WORD) < word_budget:
        b += 1
    assert int(b / TOKENS_PER_WORD) == word_budget
    return b


def sent(i: int, n_words: int) -> str:
    """One synthetic sentence of exactly n_words words, tagged S<i>."""
    return f"S{i}" + " w" * (n_words - 1) + "."


def build(n_head: int, n_mid: int, n_tail: int, n_words: int) -> tuple[str, list[str]]:
    """Text of distinct sentences; returns (text, middle sentences)."""
    sents = [sent(i, n_words) for i in range(n_head + n_mid + n_tail)]
    return " ".join(sents), sents[n_head : n_head + n_mid]


def kept_middle(out: str) -> list[str]:
    lines = out.split("\n")[1:-1]
    return [ln for ln in lines if not MARKER_RE.fullmatch(ln)]


def test_short_text_passes_through_byte_identical_with_odd_whitespace():
    text = "  hello\t\tworld \n\n  again  "
    r = truncate_text(text, 1000)
    assert r.text == text
    assert r.truncated is False
    assert (r.words_in, r.words_kept, r.words_omitted) == (3, 3, 0)


def test_exactly_at_budget_passes_through_and_one_more_word_truncates():
    wb = 50
    at = " ".join(f"w{i}" for i in range(wb))
    over = at + " extra"
    assert truncate_text(at, budget_for(wb)).truncated is False
    assert truncate_text(at, budget_for(wb)).text == at
    assert truncate_text(over, budget_for(wb)).truncated is True


def test_marker_format_is_five_whitespace_words():
    assert omission_marker(7) == "[... 7 words omitted ...]"
    assert len(omission_marker(7).split()) == MARKER_WORDS


def test_word_conservation_between_kept_marker_counts_and_fields():
    text, _ = build(15, 40, 15, 7)
    r = truncate_text(text, budget_for(150))
    assert r.truncated
    marker_total = sum(int(n) for n in MARKER_RE.findall(r.text))
    marker_words = MARKER_WORDS * len(MARKER_RE.findall(r.text))
    assert r.words_kept + marker_total == r.words_in == len(text.split())
    assert len(r.text.split()) - marker_words == r.words_kept
    assert r.words_kept + r.words_omitted == r.words_in
    assert r.words_omitted == marker_total


def test_output_length_including_markers_within_budget_plus_marker():
    for n_mid in (5, 40, 200):
        text, _ = build(15, n_mid, 15, 7)
        r = truncate_text(text, budget_for(150))
        assert len(r.text.split()) <= 150 + MARKER_WORDS


def test_head_is_verbatim_prefix_and_tail_verbatim_suffix_with_floor_counts():
    wb = 103  # floor(0.4*103) = 41 (not a round number)
    words = [f"w{i}" for i in range(400)]
    # odd whitespace inside head/tail must survive verbatim
    text = "  ".join(words[:60]) + "\n" + " ".join(words[60:340]) + "\n\n" + "\t".join(words[340:])
    r = truncate_text(text, budget_for(wb))
    n = math.floor(HEAD_SHARE * wb)
    assert n == math.floor(TAIL_SHARE * wb) == 41
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    head = text[: spans[n - 1][1]]
    tail = text[spans[-n][0] :]
    assert r.text.startswith(head)
    assert r.text.endswith(tail)
    assert len(head.split()) == n and len(tail.split()) == n
    assert r.text.split("\n")[0] == head.split("\n")[0]


def test_kept_middle_pieces_are_whole_sentences_not_word_samples():
    text, middle = build(15, 40, 15, 7)
    r = truncate_text(text, budget_for(150))
    kept = kept_middle(r.text)
    assert kept, "scenario should keep some middle sentences"
    assert all(k in middle for k in kept)
    assert all(k.endswith(".") and k.startswith("S") for k in kept)


def test_middle_split_breaks_on_newlines_and_question_or_bang_marks():
    head = " ".join(["h"] * 40)
    tail = " ".join(["t"] * 40)
    pieces = ["Q1 a b?", "Q2 c d!", "Q3 e f."]
    big = " ".join(["x"] * 60)
    r = truncate_text(
        f"{head}\n{pieces[0]} {pieces[1]}\n{pieces[2]}\n{big}\n{tail}", budget_for(100)
    )
    assert r.truncated
    # mb=20: every 3-word piece is an independent unit; the 60-word line never fits
    assert set(kept_middle(r.text)) <= set(pieces)
    assert big not in r.text


def test_cadence_is_one_fixed_stride_and_smallest_feasible_c2_case():
    # 4 middle sentences x 20 words, mb=60. c=1 costs 80 (too big); c=2 selects
    # idx 0,2: 40 words + 2 markers (10) = 50 <= 60 -> c=2.
    text, middle = build(6, 4, 6, 20)
    r = truncate_text(text, budget_for(300))
    idx = [middle.index(k) for k in kept_middle(r.text)]
    assert idx == [0, 2]


def test_cadence_stride_centred_start_and_smallest_feasible_c4_case():
    # 10 middle sentences x 4 words, mb=30. Costs: c=1 40; c=2 20+5*5=45;
    # c=3 (idx 1,4,7) 12+4*5=32; c=4 (idx 1,5,9) 12+3*5=27 -> c=4, start=(4-1)//2=1.
    text, middle = build(15, 10, 15, 4)
    r = truncate_text(text, budget_for(150))
    idx = [middle.index(k) for k in kept_middle(r.text)]
    assert idx == [1, 5, 9]
    strides = {b - a for a, b in pairwise(idx)}
    assert strides == {4}


def test_stride_is_constant_for_larger_middle():
    text, middle = build(10, 80, 10, 4)
    r = truncate_text(text, budget_for(100))
    idx = [middle.index(k) for k in kept_middle(r.text)]
    assert idx
    assert len({b - a for a, b in pairwise(idx)}) <= 1


def test_no_zero_count_markers_and_kept_words_keep_original_order():
    for n_mid in (5, 10, 40, 120):
        text, _ = build(15, n_mid, 15, 5)
        r = truncate_text(text, budget_for(150))
        assert all(int(n) > 0 for n in MARKER_RE.findall(r.text))
        out_words = MARKER_RE.sub("", r.text).split()
        it = iter(text.split())
        assert all(w in it for w in out_words)  # subsequence check


def test_single_huge_middle_sentence_is_dropped_with_one_marker():
    head = " ".join(sent(i, 4) for i in range(10))
    tail = " ".join(sent(100 + i, 4) for i in range(10))
    huge = sent(50, 500)
    r = truncate_text(f"{head} {huge} {tail}", budget_for(100))
    assert r.text.split("\n") == [head, omission_marker(500), tail]
    assert r.words_omitted == 500 and r.words_kept == 80


def test_truncate_turns_equals_truncate_text_on_newline_joined_turns():
    turns = [
        " ".join(sent(i, 6) for i in range(30)),
        "short turn",
        " ".join(sent(i, 6) for i in range(30, 60)),
    ]
    assert truncate_turns(turns, budget_for(120)) == truncate_text(
        conversation_text(turns), budget_for(120)
    )
    assert isinstance(truncate_turns(turns, 1000), Truncated)


def test_truncation_is_deterministic():
    text, _ = build(15, 40, 15, 7)
    assert truncate_text(text, budget_for(150)) == truncate_text(text, budget_for(150))
