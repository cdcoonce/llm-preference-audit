# Rung 2: hand-built features + LightGBM

Aggregates only. No prompt or response text appears here. Generated numbers live in
`results/rung-2-summary.json`; each run is a row in `results/ledger.jsonl`.

## Setup

- Rows: the 51,848 `cv` rows (46,157 prompt groups) from the grouped split. The locked test split is never
  loaded into training or scoring; `assert_cv_only` raises if a test id gets in.
- Folds: the split's own 5 folds. Outer fold k is predicted by a model trained on the other four.
  Early stopping uses an inner holdout of about 10% of the training groups, chosen by a hash of the
  group id, so a prompt never sits on both sides and fold k is never touched.
- Features: 67 per pair, from raw untruncated text only (word and char counts, turns, newlines,
  headers, bullets, code fences, bold, URLs, question marks, refusal phrases, prompt overlap; each as
  a, b, a-b, |a-b| and log-ratio; plus prompt length and turns). Model identity (`model_a`,
  `model_b`) is never used, because Kaggle's test set does not have it.
- LightGBM multiclass, learning rate 0.05, 31 leaves, min 100 rows per leaf, feature and bagging
  fraction 0.8, L2 1.0, up to 5000 rounds with 100 rounds of early stopping, seed 42, deterministic.
- `length_only`: the five word-count pair features. `full`: all 67. `full_no_swap`: all 67 without the
  a/b swap augmentation or averaging.
- Swap augmentation: every training row also appears with a and b exchanged (labels a and b flip, tie
  stays). At prediction, the row's probabilities are averaged with the mirrored prediction on the
  swapped row, so the model has no position preference by construction.

## Results

Class prior baseline (out-of-fold, each fold uses the other folds' label frequencies): 1.0973. The
full-data prior from the profile is 1.0972. "Gain vs prior" is the prior's log loss minus the model's,
with a 95% paired group-bootstrap CI (1,000 resamples of prompt groups).

| Config | Features | OOF log loss | Per-fold range | Gain vs prior [95% CI] | ECE | Best iterations (folds 0-4) | Runtime |
|---|---|---|---|---|---|---|---|
| `length_only` | 5 | 1.0591 | 1.0528 to 1.0648 | +0.0382 [+0.0357, +0.0408] | 0.0060 | 38, 46, 54, 52, 42 | 9s |
| `full` | 67 | 1.0366 | 1.0309 to 1.0425 | +0.0607 [+0.0574, +0.0640] | 0.0066 | 77, 72, 158, 120, 122 | 16s |
| `full_no_swap` | 67 | 1.0390 | 1.0350 to 1.0445 | +0.0583 [+0.0552, +0.0616] | 0.0067 | 101, 97, 66, 75, 69 | 11s |

Paired differences between configurations (positive means the second is better):

- `full` over `length_only`: +0.0225 [+0.0207, +0.0242].
- `full` over `full_no_swap`: +0.0024 [+0.0016, +0.0031].

ECE uses 10 equal-width bins of the top predicted probability between 1/3 and 1. All three models are
well calibrated (ECE under 0.007); the per-bin table is in the summary JSON.

## Gate

**Pass.** The `full` model's out-of-fold log loss (1.0366) is below the class prior (1.0972) and the
95% CI for the improvement, [+0.0574, +0.0640], excludes 0.

## What the model uses

Top 10 features by gain share, averaged over the 5 fold models (`full`):

| Feature | Gain share |
|---|---|
| n_chars_diff | 20.5% |
| n_chars_absdiff | 11.7% |
| n_newlines_logratio | 7.9% |
| prompt_words | 6.0% |
| n_words_diff | 4.0% |
| prompt_overlap_a | 2.8% |
| prompt_overlap_b | 2.8% |
| prompt_overlap_absdiff | 2.6% |
| n_chars_a | 2.6% |
| n_chars_b | 2.6% |

Length features (chars, words, newlines) carry most of the gain. The top 20 are in the summary JSON.

## Verbosity: model versus humans

Among the 35,436 cv non-ties with unequal word counts:

| | Longer response is favoured |
|---|---|
| Human labels | 61.35% (61.38% on all rows in the data profile) |
| `length_only` model (p_a vs p_b) | 96.95% |
| `full` model | 85.37% |
| `full_no_swap` model | 84.60% |

These two columns are not the same kind of number. Humans picked the longer answer 61% of the time.
The model's figure is how often it puts the higher win probability on the longer answer, not how
often it is right. Even so, a model fitted on these labels leans on length much harder than the
humans did on any single comparison: it turns a modest average tendency into a near-rule whenever
length is the strongest signal it has.

## Symmetry check

On the 310 cv rows where every per-response feature of a equals that of b:

| Config | Share where model favours a | Mean p_a - p_b | Max abs p_a - p_b |
|---|---|---|---|
| `length_only` | 0.0% | 0 | 0 |
| `full` | 0.0% | 0 | 0 |
| `full_no_swap` | 69.7% | +0.0144 | 0.0750 |

With swap averaging p_a equals p_b exactly on identical rows (so it never "favours" a). Without it
the model leans toward position a on 70% of those rows. Across all rows, the largest gap between a
raw prediction and the mirrored prediction on the swapped row is 0.123 for the swap-trained `full`
model before averaging, and 0.325 for `full_no_swap`.

## Conclusion

Cheap text-shape features beat the class prior clearly: log loss falls from 1.0973 to 1.0366, an
improvement whose 95% CI is far from zero. Length alone gets about two thirds of that gain, and the
other features add a further 0.022. Position symmetry costs nothing and helps a little: the swapped
and averaged model is 0.0024 better than the unsymmetrised one, and it has no preference for position a
by construction. The model leans on length more than the human labels do, which is the shape of
verbosity bias a later rung can compare against LLM judges. The gain is still modest: the model is
well calibrated but far from sharp, so most pairs stay near the prior.
