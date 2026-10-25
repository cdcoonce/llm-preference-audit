"""Torch-free input assembly for the rung-6 LoRA classifier.

Each row becomes `BOS + seg0 + P + seg1 + A + seg2 + B + seg3`, where P/A/B are the per-field token
ids of the adaptively truncated texts and the segments are the fixed template pieces
(`<prompt>\\n`, `\\n</prompt>\\n<response_a>\\n`, ... `Which response would a human prefer? A, B or
tie:`). Fields are tokenized separately and the template ids are concatenated, so swapping a and b
swaps the two truncated fields token-for-token.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from llm_preference_audit.deberta_inputs import water_level_cut

SEGMENTS = (
    "<prompt>\n",
    "\n</prompt>\n<response_a>\n",
    "\n</response_a>\n<response_b>\n",
    "\n</response_b>\nWhich response would a human prefer? A, B or tie:",
)
WINDOW_SLACK = 2  # tokens kept free below the limit (decided design: T - overhead - 2)


class Template(NamedTuple):
    bos: tuple[int, ...]
    segments: tuple[tuple[int, ...], ...]

    @property
    def overhead(self) -> int:
        """Tokens that are not field content: BOS plus the four template segments."""
        return len(self.bos) + sum(len(s) for s in self.segments)


def make_template(encode: Callable[[str], Sequence[int]], bos_id: int | None) -> Template:
    """`encode` must tokenize WITHOUT special tokens. `bos_id` None means the tokenizer has none."""
    segs = tuple(tuple(int(i) for i in encode(s)) for s in SEGMENTS)
    if any(not s for s in segs):
        raise ValueError("a template segment tokenized to nothing")
    return Template(() if bos_id is None else (int(bos_id),), segs)


def field_window(t: int, template: Template) -> int:
    """Token window handed to `fit_pair` so BOS + template + fields stay within `t`."""
    w = t - template.overhead - WINDOW_SLACK
    if w <= 0:
        raise ValueError(f"T={t} leaves no room for fields (template overhead {template.overhead})")
    return w


def build_lora_ids(template: Template, prompt_ids, a_ids, b_ids, limit: int) -> list[int]:
    """Assemble one sequence of at most `limit` tokens; an over-long row is hard-cut by the
    symmetric water-level rule (the same cut applies whichever response is shown first)."""
    fields = [list(prompt_ids), list(a_ids), list(b_ids)]
    budget = limit - template.overhead
    if budget < 0:
        raise ValueError(f"limit {limit} is smaller than the template overhead")
    keep = water_level_cut([len(f) for f in fields], budget)
    p, a, b = (f[:k] for f, k in zip(fields, keep, strict=True))
    s0, s1, s2, s3 = template.segments
    return [*template.bos, *s0, *p, *s1, *a, *s2, *b, *s3]
