# Rung 5: DeBERTa-v3-base fine-tuned on the labels

Aggregates only. No prompt or response text appears here. The plan and its pre-registered bar are in
`docs/plans/rung-5-deberta.md` (PR #4). Generated numbers:
- `results/rung-5-probe.json`: timing probe.
- `results/rung-5-deberta.json`: the main run, ledger row `20261024T224451-deberta_base_t1024`.
- `results/rung-5-ablation.json`: the head+tail ablation.
- `results/rung-5-debug.json`: the debug round.

## Setup

- **Model:** `microsoft/deberta-v3-base`, 3-way classification on
  `[CLS] prompt [SEP] response_a [SEP] response_b [SEP]`.
- **Window:** T = 1,024 tokens with per-row adaptive allocation through the shared `truncate_turns`.
  85.7% of cv rows kept both responses whole, and 0.08% needed a hard cut.
- **Training:** 5 folds × 2 epochs. AdamW, LR 2e-5, 10% warmup, effective batch 16, fp16 autocast with
  fp32 master weights.
  - Each example's a/b order is swapped at random each epoch.
  - The best epoch is chosen on an inner holdout of whole training groups.
  - Inference runs in both orders, averaged.
- **Platform:** Kaggle, 2× Tesla T4. Three sessions (folds 0-1, 2-3, 4) from kernel commit `5a53f15`,
  11.0 GPU-hours. Cash cost $0.
- **Scoring:** the same as rung 3. The primary is matrix calibration cross-fitted by fold. Comparators
  are scored on the same 51,848 cv rows. The locked test split was not loaded.

## Integrity bar (A): met

- The `train.csv` hash and split-manifest digest matched inside every session.
- The merged out-of-fold file covers all 51,848 cv ids exactly once, with no test ids.
- Every fold plan was group-disjoint (train / inner holdout / predicted fold).
- No fold logged a non-finite loss.
- All five folds share one commit and one config, which `merge_oof` checks.
- Hard cuts: 0.08% of rows.

## Result (B): beats the prior, not LightGBM

| | Log loss |
|---|---|
| Out-of-fold class prior | 1.0973 |
| **DeBERTa-v3-base, matrix-calibrated (primary)** | **1.0666** |
| DeBERTa, raw (swap-averaged) | 1.0753 |
| LightGBM `full` | 1.0366 |
| Haiku, matrix-calibrated | 0.9762 |

- Against the prior: **+0.031** (CI +0.027 to +0.034).
- Against LightGBM: **-0.030** (CI -0.033 to -0.027). Against Haiku: -0.090.
- Fold losses were 1.070 to 1.081.
- In every fold, epoch 2 was the best epoch and training loss was still falling (about 1.089 to 1.066).
- The predictions are close to flat: the mean top-class probability is about 0.42.

That is the pre-registered "beats the prior but not LightGBM" band, so one debug round was run.

### Debug round (fold 0, one training run per arm)

Paired group-bootstrap CIs over fold 0's 10,136 rows, measured against the main run's fold 0:

| Arm | Log loss | Change vs main |
|---|---|---|
| LR 1e-5 | 1.0782 | -0.005 (CI -0.007 to -0.002) |
| **Main: fp16, LR 2e-5, micro-batch 4** | **1.0736** | |
| LR 3e-5 | 1.0916 | -0.018 (CI -0.023 to -0.013) |
| fp32, LR 2e-5, micro-batch 2 | 1.0657 | +0.008 (CI +0.005 to +0.010) |
| *LightGBM `full`, same rows* | *1.0425* | |
| *Haiku, same rows* | *0.9739* | |

- **Learning rate:** 2e-5 is the best of the three, and the curve around it is shallow. At 3e-5 the
  model barely leaves the prior.
- **Precision:** fp32 helps by about 0.008. The run needed micro-batch 2 to fit in memory, so this
  arm also changed the batching, and each arm is a single seed: the CIs cover row sampling only, not
  run-to-run variance. fp32 also took 4.5 hours per fold against 2.2.
- **Bottom line:** no arm comes near LightGBM. The best trails it by 0.023 on the same rows.
  Per the plan, the main run's result is reported as it stands.

The DeBERTa-base fine-tune at this budget is under-trained. It is not broken: the integrity bar holds,
and both the learning-rate and precision sweeps move it only a little.

## Audit readouts (C)

These describe a model that is barely above the prior, so read them as properties of an under-trained
encoder, not as a learned preference.

- **Verbosity.** Humans chose the longer response in 61.35% of non-ties.
  - DeBERTa's calibrated share is 53.3%. On the 44,426 rows where DeBERTa saw both responses
    untruncated, it is 53.2%, against 61.4% for humans on those rows.
  - On the same rows, LightGBM matches humans (61.4%) and Haiku sits at 58.1%.
  - A model near the prior sits near 50% on any such share, so this is a sign of weak learning, not a
    length finding.
  - The length signal that LightGBM gets from one feature is not something this encoder learned in
    two epochs.
- **Position.** The first-shown response gets +0.2 points (CI +0.16 to +0.25), so the per-example
  swap augmentation works.
  - The 49.5% swap flip rate is argmax noise on near-flat predictions: the mean largest per-class
    change under swap is 0.04.
- **Agreement.** DeBERTa's p_a - p_b correlates 0.45 with LightGBM and 0.23 with Haiku. Haiku and
  LightGBM correlate 0.41 with each other.

### Head+tail ablation (decision #12, fold 0)

- With the middle 20% folded into head and tail, the loss is 1.0836, against 1.0736 for the main
  run: the middle sentences are worth **0.010** (CI 0.007 to 0.013, one seed per arm).
- The calibrated verbosity share on fold 0 is 51.5% for the main run and 50.0% for the ablation,
  against humans' 61.1%.

## Cost

- The timing probe took about 0.5 GPU-hours (three attempts: a data-path fix, an fp32-weights fix, a
  success).
- Main run 11.0, ablation 2.2, debug arms about 2.2 + 2.2 + 4.5, plus one fp32 attempt that ran out
  of memory after about 0.5 hours.
- Total about **23 GPU-hours**, inside the 40-hour gate, at $0 cash.

## What this means for the ladder

Rung 5 answers its first question (where a fine-tuned encoder lands: between the prior and the
feature model, at this budget). It cannot answer its second, whether a text-reading learner
reproduces the human length preference, because this model did not learn enough to have a
preference. That question moves to rung 6, the LoRA fine-tune of a 7-9B model. That model reads the
text and has the capacity to learn from the labels; predecessor-competition LoRA solutions reached
0.88-0.90, on a different split.
