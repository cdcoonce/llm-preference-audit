"""Tests for the shared judge/model truncation (synthetic text only)."""

from __future__ import annotations

import math
import re
from itertools import pairwise

import pytest

from llm_preference_audit import truncate as trunc_mod
from llm_preference_audit.data import conversation_text
from llm_preference_audit.profile import TOKENS_PER_WORD
from llm_preference_audit.truncate import (
    HEAD_SHARE,
    MARKER_WORDS,
    MIDDLE_SHARE,
    TAIL_SHARE,
    FieldFit,
    PairFit,
    Truncated,
    allocate_budgets,
    both_responses_whole,
    fit_pair,
    fit_to_allowance,
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


# --- head+tail ablation -------------------------------------------------------------------


def _legacy_truncate_text(text: str, budget_tokens: int) -> Truncated:
    """Frozen copy of the pre-ablation implementation (fixed 40/20/40 shares)."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    words_in = len(spans)
    word_budget = int(budget_tokens / TOKENS_PER_WORD)
    if words_in <= word_budget:
        return Truncated(text, False, words_in, words_in, 0)
    head_n = math.floor(0.4 * word_budget)
    tail_n = math.floor(0.4 * word_budget)
    head = text[: spans[head_n - 1][1]] if head_n else ""
    tail = text[spans[-tail_n][0] :] if tail_n else ""
    middle = text[len(head) : len(text) - len(tail)]
    pieces = [p for p in (s.strip() for s in SPLIT_RE.split(middle)) if p]
    counts = [len(p.split()) for p in pieces]
    chosen = set(trunc_mod._select(counts, word_budget - head_n - tail_n))
    parts = [head] if head else []
    kept = head_n + tail_n
    run = 0
    for i, piece in enumerate(pieces):
        if i in chosen:
            if run:
                parts.append(omission_marker(run))
                run = 0
            parts.append(piece)
            kept += counts[i]
        else:
            run += counts[i]
    if run:
        parts.append(omission_marker(run))
    if tail:
        parts.append(tail)
    return Truncated("\n".join(parts), True, words_in, kept, words_in - kept)


def test_default_middle_share_output_is_byte_identical_to_legacy_implementation():
    assert MIDDLE_SHARE == 0.2
    for shape in [(15, 40, 15, 7), (10, 80, 10, 4), (6, 4, 6, 20), (15, 10, 15, 4), (3, 3, 3, 3)]:
        text, _ = build(*shape)
        for wb in (20, 50, 103, 150, 300, 5000):
            b = budget_for(wb)
            assert truncate_text(text, b) == _legacy_truncate_text(text, b)
            assert truncate_text(text, b, MIDDLE_SHARE) == _legacy_truncate_text(text, b)
    turns = [
        " ".join(sent(i, 6) for i in range(30)),
        "short",
        " ".join(sent(i, 5) for i in range(40)),
    ]
    assert truncate_turns(turns, budget_for(120)) == _legacy_truncate_text(
        conversation_text(turns), budget_for(120)
    )


@pytest.mark.parametrize("wb", [100, 101, 150])
def test_middle_share_zero_is_head_tail_only_with_exactly_one_marker(wb):
    text, _ = build(15, 40, 15, 7)
    r = truncate_text(text, budget_for(wb), middle_share=0.0)
    assert r.truncated
    n = wb // 2
    lines = r.text.split("\n")
    assert len(MARKER_RE.findall(r.text)) == 1
    assert kept_middle(r.text) == []
    assert MARKER_RE.fullmatch(lines[1]) and len(lines) == 3
    words = text.split()
    assert lines[0] == " ".join(words[:n])
    assert lines[2] == " ".join(words[-n:])
    assert r.words_kept == 2 * n and r.words_omitted == len(words) - 2 * n


def test_middle_share_zero_turns_matches_text_and_non_default_share_changes_split():
    text, _ = build(15, 40, 15, 7)
    turns = [text]
    assert truncate_turns(turns, budget_for(100), 0.0) == truncate_text(
        conversation_text(turns), budget_for(100), 0.0
    )
    a = truncate_text(text, budget_for(100), 0.0)
    b = truncate_text(text, budget_for(100), 0.5)
    assert a.text != b.text


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_middle_share_raises(bad):
    with pytest.raises(ValueError):
        truncate_text("a b c", 1000, bad)
    with pytest.raises(ValueError):
        truncate_turns(["a b c"], 1000, bad)


# --- budget allocation --------------------------------------------------------------------


def test_allocation_everything_fits_returns_own_lengths():
    assert allocate_budgets(100, 200, 300, 1000) == (100, 200, 300)


def test_allocation_prompt_capped_at_quarter_window_when_responses_long():
    p, a, b = allocate_budgets(900, 800, 800, 1000)
    assert p == 250
    assert (a, b) == (375, 375)


def test_allocation_prompt_takes_slack_when_responses_short():
    p, a, b = allocate_budgets(900, 50, 50, 1000)
    assert (p, a, b) == (900, 50, 50)
    p, a, b = allocate_budgets(2000, 100, 100, 1000)
    assert (p, a, b) == (800, 100, 100)


def test_allocation_one_short_response_keeps_length_and_other_takes_rest():
    p, a, b = allocate_budgets(900, 100, 2000, 1000)
    assert p == 250 and a == 100 and b == 650
    p, a, b = allocate_budgets(900, 2000, 100, 1000)
    assert p == 250 and a == 650 and b == 100


def test_allocation_both_long_split_half_of_remainder_each_with_odd_remainder():
    assert allocate_budgets(900, 2000, 2000, 1001) == (250, 375, 375)
    assert allocate_budgets(900, 2000, 2000, 1000) == (250, 375, 375)


GRID = [0, 1, 7, 100, 249, 250, 251, 400, 1000, 3000]


def test_allocation_is_symmetric_in_a_and_b_over_grid():
    for w in (999, 1000, 1001, 512):
        for p in GRID:
            for x in GRID:
                for y in GRID:
                    pp, xx, yy = allocate_budgets(p, x, y, w)
                    assert allocate_budgets(p, y, x, w) == (pp, yy, xx)


def test_allocation_total_within_window_and_each_within_own_length_over_grid():
    for w in (999, 1000, 1001, 512):
        for p in GRID:
            for x in GRID:
                for y in GRID:
                    pp, xx, yy = allocate_budgets(p, x, y, w)
                    assert min(pp, xx, yy) >= 0
                    assert pp <= p and xx <= x and yy <= y
                    assert pp + xx + yy <= w


# --- fit_to_allowance / fit_pair ----------------------------------------------------------


def words_count(s: str) -> int:
    return 2 * len(s.split())


def overhead_count(s: str) -> int:
    return 3 * len(s.split()) + 40


def make_turns(n_sent: int, n_words: int = 6) -> list[str]:
    return [" ".join(sent(i, n_words) for i in range(n_sent))]


def test_fit_to_allowance_passes_through_unchanged_when_it_fits():
    turns = ["one two three", "four"]
    f = fit_to_allowance(turns, 100, words_count)
    assert isinstance(f, FieldFit)
    assert f.truncated.text == conversation_text(turns)
    assert f.truncated.truncated is False
    assert (f.fits, f.steps, f.tokens, f.allowance) == (True, 0, 8, 100)


def test_fit_to_allowance_steps_down_until_it_fits_with_overhead_counter():
    turns = make_turns(60)
    text = conversation_text(turns)
    allowance = 400
    assert overhead_count(text) > allowance
    f = fit_to_allowance(turns, allowance, overhead_count)
    assert f.fits and f.truncated.truncated
    assert f.steps > 0
    assert f.tokens == overhead_count(f.truncated.text) <= allowance


def test_fit_to_allowance_returns_not_fit_after_max_steps_for_impossible_allowance():
    turns = make_turns(60)
    f = fit_to_allowance(turns, 10, overhead_count, max_steps=4)
    assert f.fits is False
    assert f.steps == 4
    assert f.tokens > 10


def test_fit_pair_hard_cut_flag_and_both_responses_whole():
    prompt, a, b = make_turns(10), make_turns(10), make_turns(10)
    pf = fit_pair(prompt, a, b, 10_000, words_count)
    assert isinstance(pf, PairFit)
    assert pf.hard_cut is False and both_responses_whole(pf)
    assert not pf.prompt.truncated.truncated

    long_a = make_turns(200)
    pf = fit_pair(prompt, long_a, b, 400, words_count)
    assert pf.hard_cut is False
    assert pf.a.truncated.truncated and not pf.b.truncated.truncated
    assert not both_responses_whole(pf)
    assert pf.a.fits and pf.a.tokens <= pf.a.allowance

    pf = fit_pair(prompt, a, b, 120, overhead_count)
    assert pf.hard_cut is True


def test_fit_pair_hard_cut_considers_responses_not_only_prompt():
    prompt = ["tiny"]
    pf = fit_pair(prompt, make_turns(80), make_turns(80), 200, overhead_count)
    assert pf.prompt.fits
    assert pf.hard_cut is True
