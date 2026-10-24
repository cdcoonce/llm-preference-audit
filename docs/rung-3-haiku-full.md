# Rung 3: zero-shot Haiku judge, full cv run

Aggregates only. No prompt or response text appears here. Generated numbers live in
`results/rung-3-full.json`; the run is ledger row `20261024T153444-haiku-full` (11 batches, code at
`fba1b7a`). The 500-row pilot that gated this run is in `docs/rung-3-haiku-pilot.md`.

## Setup

Same judge, prompt, truncation and calibration as the pilot, on every labelled `cv` row:

- 51,848 rows, each judged in both orders: 103,696 calls to `claude-haiku-5-5` (effort `low`,
  JSON-schema output, no sampling parameters) through 11 Message Batches of up to 10,000 requests.
- The locked test split was not sent and not scored.
- Primary score: matrix scaling on the log stated probabilities, cross-fitted by fold (fold k is
  calibrated on the other four), pre-registered at the pilot.
- Comparators: the out-of-fold class prior and rung 2's out-of-fold LightGBM `full` on the same rows.
  Gains are the comparator's log loss minus the judge's, with 95% paired group-bootstrap CIs over
  prompt groups.

## Results

| | Log loss |
|---|---|
| Out-of-fold class prior | 1.0973 |
| Haiku stated probabilities, A/B order only | 1.4315 |
| Haiku stated probabilities, swap-averaged | 1.3154 |
| Haiku, temperature-calibrated | 1.0530 |
| **Haiku, matrix-calibrated (primary)** | **0.9762** |
| LightGBM `full` (rung 2) | 1.0366 |

- Gain over the prior: **+0.121** (CI +0.117 to +0.125).
- Gain over LightGBM: **+0.060** (CI +0.056 to +0.065). A zero-shot judge with a three-class
  calibration layer beats the 67-feature model trained on the same labels.
- Haiku's stated probabilities are badly overconfident on their own: raw, they score worse than the
  prior. A single temperature recovers about a third of the gain; matrix scaling, which also corrects
  how the judge spreads mass across a, b and tie, recovers the rest.
- 103,601 of 103,696 calls returned valid JSON. 88 were refused (0.085%) and 7 did not parse; those
  calls fell back to the class prior, touching 0.11% of rows. Without those rows the loss is 0.9761.
  The stated `winner` matched the argmax of the stated probabilities in 94.1% of calls.

### Position bias

- Swapping the order changes the judge's argmax on 18.6% of rows (mean largest per-class change
  0.12).
- The first-shown response gets a mean stated probability of 0.475 against 0.446 for the second:
  **+2.9 points** (CI +2.8 to +3.1). Among non-tie verdicts the first-shown response wins 52.1%.
  Humans' first-position effect over all rows is about +0.7 points (a 34.91% vs b 34.19%).
- Swap-averaging removes the judge's position preference from the final probabilities by
  construction. A single-order judge would carry it, which is part of why the A/B-only score is so
  much worse.

### Verbosity

On the 35,436 non-ties where the two responses differ in length, humans chose the longer one 61.35%
of the time. The judge's calibrated share (mean p_longer / (p_longer + p_shorter)) is **57.87%**:
**-3.5 points** (CI -3.9 to -3.1). The uncalibrated swap-averaged share is 57.43%, so calibration
did not create the gap.

| Length gap (abs log1p word ratio) | Rows | Humans | Haiku |
|---|---|---|---|
| 0.00 - 0.17 | 7,083 | 52.2% | 51.2% |
| 0.17 - 0.38 | 7,091 | 57.4% | 54.4% |
| 0.38 - 0.65 | 7,088 | 62.6% | 56.9% |
| 0.65 - 1.17 | 7,084 | 66.0% | 60.5% |
| 1.17 - 6.42 | 7,090 | 68.6% | 66.3% |

Haiku's preference rises with the length gap, as humans' does, but sits below the human rate in every
quintile, most clearly at moderate gaps. So the judge is not length-blind: it under-weights length
relative to Arena voters. That is the opposite direction from the common "LLM judges prefer longer
answers" claim, and the first cast in this audit that departs from the human labels. The trained
LightGBM matched humans (61.40% vs 61.35%).

Caveat: the judge sees truncated text (1.9% of rows lost some words, 3.3% of words overall). The
omission markers report how much was cut, and so few rows are affected that truncation cannot account
for a 3.5-point gap.

## Cost

154.8M input and 7.5M output tokens (about 73 per call), **$9.62** at batch prices, against a
pre-submit gate estimate of $11.49 (the 99% input bound from 2,000 counted rows plus twice the
pilot's output rate). Judge spend to date: $9.72 of the $40 allotment.
