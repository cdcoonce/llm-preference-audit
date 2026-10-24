# Plan: rung 5, DeBERTa-v3 on Kaggle GPUs

Status: **draft for review** (2026-10-24). Nothing here has run. Numbers marked *measured* come from
local aggregate measurements on cv rows; numbers marked *unverified* come from third-party sources
or priors and are checked by the timing probe before any full run.

## What this rung answers

Rungs 2 and 3 gave two casts that disagree with each other:

| Cast | cv log loss | Calibrated verbosity share (humans 61.35%) |
|---|---|---|
| LightGBM, 67 length/structure features | 1.0366 | 61.40% (matches humans) |
| Haiku, zero-shot + calibration | **0.9762** | 57.87% (-3.5 points) |

DeBERTa is the first model that both **learns from the labels** and **reads the text**. The rung asks:

1. Where does a fine-tuned encoder land between the feature model and the zero-shot judge?
2. Does a text-reading learner reproduce the human length preference, as LightGBM does, or depart
   from it as Haiku does? A learner trained with cross-entropy on these labels should drift toward
   the human rate; if it does not, the departure is about what the text lets it see.

The first is a leaderboard question; the second is the audit question and the reason the rung
exists.

## Fixed inputs (identical to rungs 2 and 3)

- **Split:** the committed grouped split. The notebook rebuilds the manifest with
  `llm_preference_audit.split.build_manifest` and refuses to run unless its digest equals
  `splits/split_manifest.sha256`. Training, early stopping and calibration only ever see `cv` rows;
  the locked test split is never loaded into a model or scored.
- **Data:** inside Kaggle, the competition's attached `train.csv`, which is byte-identical to the
  Hugging Face copy (sha256 `0692154a…`, checked in profiling). The notebook checks the hash before
  use. Kaggle's `test.csv` is attached by the competition source but is never opened
  (decision #10; the notebook asserts it is not read).
- **Code:** the notebook installs this repo at a pinned commit (`pip install
  git+https://github.com/cdcoonce/llm-preference-audit@<sha>`), so truncation, split and labels are
  the same functions, not copies. The SHA goes in the ledger row.
- **Scoring:** done locally, not on Kaggle. The notebook outputs out-of-fold probabilities only
  (id + p_a / p_b / p_tie, plus each order's raw probabilities); the same `scoring`, `calibrate` and
  `bias` functions score them. The comparators are the out-of-fold prior, LightGBM `full` and Haiku
  on the same 51,848 rows.

## Input budget: the one real design decision

DeBERTa reads a fixed window; Haiku read up to about 4,000 estimated tokens per row. *Measured* on
3,000 cv rows with the deberta-v3-base tokenizer:

- Untruncated pair length (prompt + a + b): p50 507, p90 1,224, p99 3,260 tokens. 85% of rows fit
  in 1,024 tokens, 94% in 1,536.
- Tokens per word: prompts 1.45, responses 1.36 (code-fenced text ~1.9; mostly non-ASCII text ~5.9).
- With fixed 20/40/40 field shares, scaled down so that under 1% of rows overflow: T=1,024
  truncates **61% of rows and drops 24% of words**; T=1,536 truncates 22% and drops 6%.

That is the confound: Haiku saw 1.9% of rows truncated, so it saw both full lengths almost
everywhere. A model that sees heavily cut responses can only read length through the
`[... N words omitted ...]` counts, and encoders are poor at reading numbers. A verbosity gap could
then be an artefact of the window, not of what the model learned.

**Plan:**

1. **Adaptive allocation.** Each field still goes through the shared `truncate_turns` (head 40% /
   middle 20% whole sentences / tail 40%, with markers). Only the per-field budgets change, set
   per row by water-filling in DeBERTa tokens:
   - The usable window is W = T − 4 (for [CLS] and three [SEP]).
   - The prompt gets `min(L_p, max(0.25·W, W − L_a − L_b))`: at most a quarter of the window, unless
     the responses need less than the rest.
   - The responses share the remainder R. A response that needs at most R/2 keeps its full length
     and the other gets what is left; otherwise each gets R/2. The split is therefore symmetric in
     a and b by construction.
   - Each allowance converts to a `truncate_turns` budget through that field's own tokens per word,
     with a safety factor stepped down 0.05 at a time (at most 10 steps) until the field fits.
2. **Window: T = 1,024.** *Measured* on the same 3,000 rows:

   | T | Rows truncated | Words omitted | Both responses untruncated | Hard-cut rows |
   |---|---|---|---|---|
   | 768 | 28.4% | 11.1% | 72.8% | 0.13% |
   | **1,024** | **15.2%** | **5.7%** | **85.7%** | **0.03%** |
   | 1,536 | 5.9% | 2.1% | 94.9% | 0% |
   | 1,024, fixed 20/40/40 shares | 60.9% | 23.8% | 42.7% | 0.9% |

   Adaptive allocation at 1,024 nearly matches fixed shares at 1,536 (22% of rows truncated, 6% of
   words), at about two-thirds of the sequence length. The allocation loop costs about 0.4 ms per
   row, so about 25 s for the full data. Move to 1,536 only if the probe shows spare GPU budget.
3. **Hard cut:** rows still over T after truncation are cut by the tokenizer
   (`truncation="longest_first"`), counted and reported. That was 0.03% of the measured rows.
4. **Pre-registered verbosity comparison across casts:** the headline compares humans, LightGBM,
   Haiku and DeBERTa **on the rows where DeBERTa saw both responses untruncated**, so all casts see
   full length information on the same rows. The all-rows readout is reported too, plus the
   truncated subset, so the window effect is itself measured.

## Model and training

| Choice | Plan | Why |
|---|---|---|
| Model | `microsoft/deberta-v3-base` (184M) | `large` is ~3x the compute; base fits 5 folds in the free quota. Large only if the probe shows headroom. |
| Input | `[CLS] prompt [SEP] response_a [SEP] response_b [SEP]` | Standard pair encoding; segments in a fixed order. |
| Head / loss | 3-way classification, cross-entropy | Directly optimises the scored quantity. |
| Precision | fp16 with dynamic loss scaling, grad clip 1.0 | T4 has no bf16; DeBERTa-v3 is known to hit fp16 NaN, so any non-finite loss stops the run (it is not skipped). |
| Optimiser | AdamW, LR 2e-5, 10% warmup, linear decay, weight decay 0.01 | Common DeBERTa fine-tuning defaults. |
| Epochs | 2, keep the best checkpoint on the inner holdout | |
| Batch | effective 16 (micro-batch and accumulation set by memory at T) | |
| Inner holdout | 10% of each fold's training groups, by hash of the group id | Same rule as LightGBM's early stopping; fold k is never used for its own training or selection. |
| Position | random a/b swap per example per epoch (p=0.5); inference in both orders, averaged | Same treatment as LightGBM's swap augmentation at no extra training cost; the two orders give the position readouts. |
| Seed | fixed per fold, recorded | |

## Compute plan

*Unverified* platform facts (from third-party pages dated October 2026 and the Kaggle CLI 2.2.4
help): about 30 GPU hours per week, 12-hour sessions, T4 x2 (16 GB each, no bf16) as the default
accelerator through `kaggle kernels push --accelerator NvidiaTeslaT4`. The probe confirms them.

1. **Timing probe (one short session):** 200 training steps and 2,000 inference rows at the chosen T,
   on one GPU. It records samples per second, peak memory and whether fp16 stays finite. It projects
   GPU hours for 5 folds x 2 epochs plus inference.
2. **Go to full training only if** the projection is at most 40 GPU hours (about 1.5 weeks of
   quota). Otherwise step down in this order: 1 epoch, then T=768, and re-probe.
3. **Layout:** one fold per GPU, two folds per session (T4 x2), each session under the 12-hour limit,
   so 5 folds take 3 sessions. Each fold writes its out-of-fold file as it finishes, so a lost
   session costs one fold, not the run.

Rough *unverified* expectation: 2-4 GPU hours per fold-epoch at T=1,024 on a T4, so 20-40 GPU hours
for 5 folds x 2 epochs. The probe replaces this guess.

## Execution path (API, no manual notebook editing)

- `kaggle/rung5_deberta/` in the repo: the kernel script plus `kernel-metadata.json` with
  `is_private: true`, `enable_gpu`, `enable_internet` (to install the pinned repo and download the
  model), `competition_sources: ["llm-classification-finetuning"]`, and the fold list as a parameter.
- Push with `kaggle kernels push`, watch with `kaggle kernels status`, fetch outputs with
  `kaggle kernels output` into gitignored `data/predictions/`.
- Outputs are id + probabilities and training-loss curves only, never text or weights. The local
  leak check runs on everything fetched.
- `scripts/score_oof.py` (new, generic): scores any out-of-fold file the same way rung 3 was scored
  and appends one ledger row (`cost_usd` 0, plus `gpu_hours`, `kernel_version` and the pinned SHA).

## Go / no-go bar (pre-registered)

**A. Integrity: all must hold, or the run is void and is fixed and re-run.**

- The manifest digest and the `train.csv` hash match inside the notebook.
- The out-of-fold file covers all 51,848 cv ids exactly once and no test id.
- No fold's training or selection used its own rows (checked from logged group ids per fold).
- No non-finite loss in any fold's log.
- Truncation came from `truncate_turns` at the pinned SHA, and hard-cut rows are ≤ 1.5%.
- The ledger row records the SHA, config hash, kernel version and GPU hours.

**B. Performance: decides what the result means, not whether it is reported.**

| Outcome (matrix-calibrated, cross-fitted, same as Haiku) | Reading | Action |
|---|---|---|
| Does not beat the prior (CI includes 0) | Broken run | No-go: debug (one fp32 fold, LR 1e-5 / 3e-5) and re-run |
| Beats the prior but not LightGBM `full` 1.0366 (CI) | Suspicious: a text model losing to length features | One debug round as above, then report whatever holds |
| Beats LightGBM, CI excludes 0 | **Go**: valid learned-model cast | Report; proceed |
| Also beats Haiku 0.9762 (CI excludes 0) | Strong: learning beats zero-shot | Report; raises the bar for rung 6 |

The raw (uncalibrated) loss is reported beside the calibrated one. A trained model should need little
calibration; a large gap would itself be a finding.

**C. Audit readouts: no pass/fail; pre-registered so they cannot be chosen after the fact.**

- Calibrated verbosity share vs humans: on untruncated-for-DeBERTa rows (headline), all rows, and
  truncated rows; by length-gap quintile.
- Position: the swap flip rate and P(first) - P(second).
- Agreement structure: correlation of DeBERTa's and Haiku's out-of-fold p_a - p_b, and of each with
  LightGBM, to show whether the text learner sits nearer the feature model or the judge.

**D. Budget.**

- $0 cash. At most 40 Kaggle GPU hours for the full run plus probe.
- Over that, stop and re-plan; there is no silent step-down after the probe.

**E. Gate into rung 6 (draft).**

- Rung 6 spends the $100 GPU allotment, so it needs Charles's sign-off.
- It needs rung 5's integrity bar met.
- A LoRA model justifies being the "best learned model" only if it beats max(Haiku, DeBERTa) with a
  CI excluding 0. Predecessor-competition Gemma-2-9B LoRA scores of 0.88-0.90 suggest that is
  realistic.
- Those are scores on a different split and leaderboard, so they are not a target.

## Decisions for Charles

1. **Window T:** 1,024 with adaptive allocation (recommended: 85.7% of rows keep both responses
   whole) vs 1,536 (94.9%, for roughly 1.5-2x the compute).
2. **Base vs large:** base (recommended); large only if the probe leaves room.
3. **Head+tail ablation** (decision #12): run it here on one fold, with the middle 20% folded into
   head and tail (recommended, cheap), or defer it.
4. **Primary score:** matrix-calibrated, cross-fitted, the same as Haiku (recommended), or the raw
   loss.
5. **Notebook privacy:** private kernels (recommended), with the code public in this repo.

## Work breakdown

1. Adaptive budget allocator in `truncate.py` (`allocate_budgets(lengths, T)` plus a tokenizer-aware
   wrapper), test-first and mutation-checked. Tests cover a/b symmetry, never exceeding W, short rows
   untouched and the safety-step loop. Measured on all cv rows before training.
2. `scripts/score_oof.py` plus tests (synthetic OOF files).
3. Kaggle kernel script and metadata; a dry run on CPU with a tiny model on 200 rows.
4. Timing probe on Kaggle.
5. Five folds over three sessions.
6. Scoring, write-up (`docs/rung-5-deberta.md`), ledger row, vault note.

## Risks

- **fp16 instability on DeBERTa-v3:** it stops the fold. Fallback is fp32 on one fold to tell a bug
  from instability.
- **Kaggle quota or session limits differ from the sources:** the probe catches it.
- **Truncation confound:** reduced by adaptive allocation (85.7% of rows keep both responses whole)
  and handled by the pre-registered untruncated subset, which therefore covers most rows rather than
  a selected minority. That subset still skews short, so it is compared within itself across casts,
  never against all-rows figures.
- **No published DeBERTa scores for this competition were found;** the performance bar is anchored
  on this repo's own comparators instead.
