# Rung 3 pilot: zero-shot Haiku judge

Aggregates only. No prompt or response text appears here. Generated numbers live in
`results/rung-3-pilot.json`; the run is ledger row `20261024T022415-haiku-pilot`
(batch `msgbatch_01MsF2dsd3pvLEtkifXCSX5V`, code at `0fe1315`).

## Setup

- Rows: 500 `cv` rows (497 prompt groups, all five folds), picked by a salted hash of the row id.
  The locked test split is never sent; `cross_fit` raises if a non-cv row reaches calibration.
- Text: the Hugging Face copy, through the shared `truncate_turns` (prompt 1,000 and each response
  1,500 estimated tokens, head 40% / middle 20% as whole sentences / tail 40%, with
  `[... N words omitted ...]` markers). Only 1.6% of the pilot rows needed any truncation.
- Judge: `claude-haiku-5-5` through the Message Batches API, effort `low`, JSON-schema output
  `{winner, p_a, p_b, p_tie}`. No sampling parameters (rejected since 2026-10-21), no logprobs.
  Every row is judged in both orders (A/B and B/A); the swapped answer is mapped back and the two
  are averaged.
- Calibration (pre-registered primary): matrix scaling, a multinomial logistic regression on the
  log stated probabilities, cross-fitted by fold (fold k is calibrated on the other four).
  Temperature scaling is reported alongside.
- Comparators on the same 500 rows: the out-of-fold class prior (the gate) and rung 2's
  out-of-fold LightGBM `full`.

## Results

| | Log loss |
|---|---|
| Out-of-fold class prior | 1.0965 |
| Haiku stated probabilities, A/B order only | 1.6153 |
| Haiku stated probabilities, swap-averaged | 1.3414 |
| Haiku, temperature-calibrated | 1.0584 |
| **Haiku, matrix-calibrated (primary)** | **1.0049** |
| LightGBM `full` (rung 2) | 1.0387 |

- **Gate passed.** Gain over the prior +0.092 (95% paired group-bootstrap CI +0.049 to +0.130).
- Against LightGBM the judge is ahead by 0.034, but the CI (-0.014 to +0.078) includes 0: 500 rows
  cannot separate the two.
- The stated probabilities are badly overconfident on their own (worse than the prior). Calibration
  is what makes the judge usable, and the class-bias part of it matters: temperature alone recovers
  less than half of the gain.
- 1,000 of 1,000 calls returned valid JSON; no refusals, so no prior fallback was used. The stated
  `winner` matched the argmax of the stated probabilities in 93.6% of calls.

### Position bias

- Swapping the order changes the judge's argmax on 15.0% of rows (mean largest per-class change
  0.11).
- Across all 1,000 calls, the first-shown response gets a mean stated probability of 0.474 against
  0.450 for the second: +2.3 points (CI +1.1 to +3.6). Humans' first-position effect over all rows
  is about +0.7 points (a 34.91% vs b 34.19%). Swap-averaging removes the judge's preference from the
  final probabilities by construction.

### Verbosity

On the 345 pilot non-ties where the two responses differ in length, humans chose the longer one
62.0% of the time. The judge's calibrated share (mean p_longer / (p_longer + p_shorter)) is
**56.9%**: -5.1 points (CI -9.6 to -0.4). The uncalibrated swap-averaged share is the same 56.9%,
so this is not a calibration artefact, and almost no rows were truncated, so it is not the
omission markers either.

Read with care: this is a 500-row pilot and the CI's upper end sits near 0. If it holds in the full
run, it is the audit's first divergence from the human labels, and in the opposite direction from
the usual "LLM judges prefer longer answers" story. The trained LightGBM matched humans (61.40% vs
61.35%). The per-length-gap buckets hold about 69 rows each and are too noisy to read.

## Cost

1,493,530 input and 70,080 output tokens (about 70 per call: Haiku spent 0 thinking tokens at effort
`low`), **$0.092** at batch prices. The real input count was about twice the local words x 1.33
estimate, because the judge prompt, the schema and Haiku's tokenizer cost more than the profile's
rule of thumb.

At the measured rates, a full run of all 51,848 cv rows in both orders is about 155M input and
7.3M output tokens, roughly **$9.6**, inside the $40 judge allotment.
