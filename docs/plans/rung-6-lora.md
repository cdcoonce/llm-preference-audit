# Plan: rung 6, LoRA fine-tune of a 7-9B LLM

Status: **draft for Charles's sign-off** (2026-10-25). Nothing has been spent. Numbers marked
*measured* come from local aggregates on cv rows. Numbers marked *estimated* come from first-principles
throughput and current list prices, and the probe replaces them before the main spend.

## What this rung answers

| Cast | cv log loss | Calibrated verbosity share (humans 61.35%) |
|---|---|---|
| LightGBM, length/structure features | 1.0366 | 61.40% (matches humans) |
| DeBERTa-v3-base, fine-tuned | 1.0666 | 53.3% (under-trained, near 50% by default) |
| Haiku, zero-shot + calibration | **0.9762** | 57.87% (-3.5 points) |

Rung 5 could not answer the audit question because DeBERTa learned too little to have a preference.
Rung 6 needs a **competent text-reading learner** to answer it:

1. Does a model that reads the text and learns from the labels reproduce the human length
   preference, as LightGBM does, or under-weight it, as Haiku does?
2. Where does a 7-9B LoRA model land relative to Haiku? And as decision #7 requires, what does the
   Kaggle submission score? The leaderboard score is reported, not targeted (decision #1).

## Base model: Gemma-2-9B (recommended), Qwen3-8B as fallback

| | Gemma-2-9B | Qwen2.5-7B | Qwen3-8B |
|---|---|---|---|
| Licence | Gemma Terms of Use (custom; last modified 2026-04-01) | Apache-2.0 | Apache-2.0 |
| HF access | **Gated** (click-through) | Open | Open |
| On Kaggle Models (offline submission) | Yes (`google/gemma-2`) | Yes | Yes |
| Evidence on this task | **Main backbone of every top LMSYS solution**; single-model val about 0.88-0.90 | None found | None found |
| Attention | Logit softcapping; fast-kernel support unconfirmed (may need eager, est. +20-30% time) | Standard (FlashAttention / SDPA) | Standard |

**Why Gemma-2-9B:** the rung needs a learner that is actually good at this task, and Gemma-2-9B is the
only candidate with documented evidence of that. The predecessor's 3rd place called it "easily the
best" backbone. Choosing an untested model would risk repeating rung 5, a cast too weak to answer
the question.

**Licence check (the open item from the grill):**

- **Training and results:** the Gemma Terms put no restriction on training with CC BY-NC data, on
  commercial use, or on publishing results. Outputs are not Model Derivatives, and Google claims no
  rights in them.
- **The repo:** it holds code only and never contains Gemma weights or adapters, so it needs nothing
  beyond its MIT licence.
- **The adapter:** a LoRA adapter is very likely a "Model Derivative". It stays private (a private
  Kaggle dataset plus private storage). If it is ever published, it must carry the Gemma Terms, the
  use restrictions and a Notice file. That is a notice obligation, not a blocker.
- **Prohibited Use Policy:** nothing in it touches research, benchmarking or publication.
- **Needs Charles:** accept the Gemma terms on Hugging Face (instant click-through). Kaggle Models
  hosts Gemma-2 too, but the training box downloads from Hugging Face.

**Fallback, Qwen3-8B:** if Charles prefers not to accept the Gemma terms. It is Apache-2.0, not gated
and faster, but carries a higher risk of a weak cast. Qwen2.5 and Qwen3 tokenize identically on this
data (*measured*).

**Not chosen without checking:** Google moved Gemma 4 to Apache-2.0 in March 2026 (secondary sources).
If a 7-12B Gemma 4 model exists and has results on this task, it may beat both. That is unverified
and left as a question, not a plan change.

## Fixed inputs (same as rungs 2-5)

- **Split:** the same grouped split. The training box checks the split-manifest digest and refuses to
  run if it differs. Training, early stopping and calibration use cv rows only. The locked test split
  stays locked until the end-of-ladder evaluation.
- **Data:** the Hugging Face copy (byte-identical to Kaggle's). It is uploaded to a **private** cloud
  volume for processing and never published; competition rule 4b forbids redistribution, not private
  processing. Kaggle's `test.csv` appears only in the submission notebook (decision #10).
- **Truncation:** the same `truncate_turns`, with the same adaptive per-row allocation as rung 5.
- **Scoring:** local, through `scripts/score_oof.py`, against the prior, LightGBM, Haiku and DeBERTa on
  the same 51,848 rows.

## Input window: T = 2,048

*Measured* on 3,000 cv rows with the Qwen tokenizer. Gemma's tokenizer is gated, so the probe
re-measures it.

| T | Rows truncated | Both responses whole | Words omitted | Mean input tokens |
|---|---|---|---|---|
| 1,024 | 16.4% | 84.7% | 6.2% | 564 |
| 1,536 | 6.4% | 94.5% | 2.3% | 622 |
| **2,048** | **3.1%** | **97.4%** | **1.1%** | **646** |
| 3,072 | 1.2% | 98.9% | 0.3% | 668 |

Training cost follows the mean length because batches are padded dynamically. So T = 2,048 costs only
about 15% more than 1,024 and keeps both responses whole on 97% of rows. The truncation confound that
weighed on rung 5 largely disappears. Top LMSYS solutions trained at 1,024-1,800 and ran inference up
to 2,000-4,000.

## Model and training

| Choice | Plan | Source / reason |
|---|---|---|
| Head | `Gemma2ForSequenceClassification`, 3 classes, cross-entropy | Scored quantity |
| Input | `<prompt>…</prompt> <response_a>…</response_a> <response_b>…</response_b>` plus a short question (35 template tokens) | Measured overhead |
| LoRA | r = 16, alpha = 32, dropout 0.05, all linear layers; classification head trained fully | Rank ≥ 16 recommended (LMSYS 16th); top teams used 16-64 |
| Precision | bf16 weights and LoRA (H100), no 4-bit in training | bf16 on Hopper; avoids QLoRA quality loss |
| Optimiser | AdamW, LR 1e-4, 5% warmup, cosine decay, weight decay 0.0 on LoRA | Mid-range of the 5e-5 to 2e-4 reported |
| Epochs | **1** | Top single models used 1-2; budget |
| Batch | Effective 16 (micro-batch from the probe), gradient checkpointing | |
| Position | Random a/b swap per example (p = 0.5); both-order inference, averaged | Rungs 2 and 5 do the same; swap TTA was worth +0.003 to +0.007 in LMSYS |
| Selection | Inner holdout (whole groups) evaluated at 50% and 100% of the epoch; keep the better | Fold k never seen |
| Attention | Probe compares SDPA with eager on 200 rows. Use SDPA only if its logits match eager within 1e-3; otherwise eager | Softcapping support is unconfirmed |

## Compute, provider and cost

**Provider: Modal (recommended).**

- **Reproducible:** a Python-defined container runs a pinned repo commit (decision: API paths, no
  manual consoles). Billing is per second.
- **Cost:** H100 is $3.95/h, about the same as RunPod Secure ($3.99). Modal's free tier includes
  $30/month of compute (documented), which lowers the cash cost.
- **Cheaper alternative:** RunPod Community H100 at $2.69/h. It is cheaper but less reliable, and
  needs pod API scripting.
- **Rejected:** A100 is about half the speed for less than half the saving. L40S has the least memory
  headroom.

**Estimated cost on 1× H100.** Tokens are *measured*; throughput is *estimated*: 5.5-7.3k training
tokens/s and 15-20k inference tokens/s, plus 30% for Gemma if eager attention is needed.

| Stage | Tokens | H100 hours | Cost at $3.95/h |
|---|---|---|---|
| G0 probe: 200 steps, 2k-row inference, SDPA/eager check | | 0.5 | $2 |
| G1 fold 0: train + both-order predict | 26.8M + 13.4M | 1.4-2.1 | $6-8 |
| Folds 1-4 | 4 × (26.8M + 13.4M) | 5.6-8.4 | $22-33 |
| Submission adapter, trained on all cv rows | 33.5M | 1.6-2.3 | $6-9 |
| Overheads (image, downloads, idle) | | about 15% | $5-8 |
| **Total** | | **about 10-14** | **$41-60** |

The hard ceiling is the $100 GPU allotment (decision #20). The $30 free credit, if applied, comes off
the cash cost. Judge spend so far is $9.72 of $40.

**Kaggle submission** (free, Kaggle T4 × 2, 9-hour limit, internet off):

- Gemma-2-9B from Kaggle Models, with the submission adapter as a private Kaggle dataset.
- Inference in 8-bit (bitsandbytes) on about 25,000 test rows, both orders, sorted by length, half the
  rows per GPU. This is the setup the LMSYS 5th place documented fitting the 9-hour limit.
- Verified with a timed 1,000-row dry run on Kaggle before the first submission. The leaderboard score
  is reported, not a gate.

## Go / no-go bar (pre-registered)

**Each spend gate below is a hard stop. Moving past a stop needs Charles.**

**G0, probe (at most $5).** All must hold before G1:

- loss is finite;
- memory fits at the chosen micro-batch;
- the SDPA-vs-eager check is decided;
- the measured throughput projects a total of at most $70.

Otherwise stop and re-plan (smaller T, Qwen3-8B, or QLoRA).

**G1, fold 0 (at most $10)** decides whether to spend on the other four folds:

| Fold-0 log loss (swap-averaged, raw) | Reading | Action |
|---|---|---|
| ≤ 1.00 | Competent learner (LightGBM 1.0425 and Haiku 0.9739 on fold 0) | **Go**: folds 1-4, then the submission adapter |
| 1.00 to 1.0425 | Weak, like rung 5 | Stop; report; Charles decides |
| > 1.0425 (worse than LightGBM) | Likely broken | Stop; debug one round (LR 5e-5 / 2e-4); no further folds without Charles |

**A. Integrity** (all must hold, or the run is void):

- Same split and data checks as rung 5. Every cv id appears exactly once out of fold.
- Every fold plan is group-disjoint. No non-finite loss.
- One commit and one config across all folds, with every fold log checked against the requested
  config.
- No data, adapter or weights in the public repo.
- The ledger row records the commit, config hash, GPU hours and the real `cost_usd`.

**B. Performance** (matrix-calibrated, cross-fitted, same as rungs 3 and 5):

| Outcome | Reading |
|---|---|
| Beats Haiku 0.9762, CI excludes 0 | Best learned model; the rung-6 headline |
| Between LightGBM and Haiku | Valid learned cast; reported as is |
| Not better than LightGBM | Reported; the learned-model cast is then LightGBM |

**C. Audit readouts** (pre-registered, no pass/fail):

- Calibrated verbosity share on the rows where both responses are whole (97%), all rows, and by
  length-gap quintile. Every cast is read on the same rows: humans, LightGBM, Haiku, DeBERTa, LoRA.
- Position (swap flip rate, P(first) - P(second)).
- Agreement correlations with every cast.
- The headline question: **does a competent text learner sit at about 61% (learns the human preference)
  or below (shares Haiku's under-weighting)?**

**D. Budget.** At most $100 of GPU in total. The runner reads the ledger and refuses any job whose
estimate would push cumulative GPU spend past $85, leaving room for the submission adapter.

## Decisions for Charles

1. **Base model:** Gemma-2-9B (recommended; accept its terms on Hugging Face) or Qwen3-8B (Apache-2.0,
   no evidence on this task).
2. **Provider:** Modal H100 (recommended; needs a Modal account and API token set up by Charles) or
   RunPod Community H100 (cheaper, less reliable).
3. **Spend:** sign off G0 + G1 (at most $15) now, with the folds-1-4 + submission spend (about $30-45)
   automatic if G1 passes. Or ask to sign off again after G1.
4. **Window:** T = 2,048 (recommended) or 1,536 (about 4% cheaper, 94.5% of rows whole).
5. **Submission adapter:** train one adapter on all cv rows (recommended, about $8). Or submit the
   fold-0 adapter and save the spend.

## Work breakdown

1. `scripts/lora/` (Modal app or RunPod job). It reuses `deberta_inputs` for fold plans, the OOF
   format, `merge_oof`, `validate_fold_parts` and the config check. Built test-first with a CPU dry
   run on a tiny Gemma2 config.
2. Ledger-aware spend guard and cost reporting from provider usage.
3. G0 probe, then G1 fold 0, then folds 1-4 (one container per fold, resumable), then merge and score.
4. The submission adapter, the Kaggle inference notebook with a timed dry run, and the submission.
5. Write-up, ledger, vault note.

## Risks

- **Throughput is estimated, not measured:** G0 measures it before any real spend.
- **Gemma-2 softcapping under SDPA is unconfirmed:** the probe checks numerical agreement with eager.
- **Licence:** the Gemma terms were revised in April 2026 and the diff wasn't found. The plan never
  distributes Gemma derivatives, so the revision cannot block it, but it must be re-read before any
  adapter is ever published.
- **Kaggle inference time:** 9B at 8-bit with swap TTA on about 25k rows is documented as fitting 9
  hours. A timed dry run confirms it.
- **The repeat of rung 5 (an under-trained cast):** G1's ≤ 1.00 bar catches it after about $10, not $50.
