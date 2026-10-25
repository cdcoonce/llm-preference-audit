# Plan: rung 6b, free Qwen3 LoRA on Kaggle T4s (fold-0 scale)

Status: **draft for review** (2026-10-25). Nothing has run.

This plan replaces the paid Gemma run (`rung-6-lora.md`) *for now*. It needs no Hugging Face account,
no Modal and no spend: only Kaggle's free GPUs and a model Kaggle hosts. If it produces a competent
learner, the paid run may not be needed. If not, it tells us so for free.

## Why

| Cast | cv log loss | Submittable on Kaggle? |
|---|---|---|
| LightGBM `full` | 1.0366 | Yes (CPU notebook built, not yet submitted) |
| LightGBM + DeBERTa stack | 1.0291 | Needs a DeBERTa all-rows model |
| Haiku, calibrated | 0.9762 | **No**: internet is off in submissions |
| LightGBM + Haiku stack | 0.9524 | No |

The two best results need an API call, so they cannot be submitted. The submittable lever is a
fine-tuned LLM. This rung asks:

1. Does a small open LLM fine-tuned for free beat LightGBM on fold 0?
2. If it is competent, does it reproduce the human length preference (61.35%) or under-weight it as
   Haiku does (57.87%)? That is rung 6's audit question, answered for free.

## Model: Qwen3-1.7B first

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| Kaggle handle | `qwen-lm/qwen-3/transformers/1.7b` (4.1 GB) | `.../4b` (8.1 GB) |
| Licence | Apache-2.0 (Kaggle metadata) | Apache-2.0 |
| fp16 memory on a 16 GB T4 | about 4 GB of weights; room to spare | about 8 GB; micro-batch 1, probably fits |
| Fold-0 training, *estimated* | 3-8 h on one T4 (about half with both) | 8-20 h on one T4; 4.5-11 h on both |

- **Why 1.7B:** 4B's fold 0 may not fit the 12-hour session even across both GPUs. 1.7B fits with a
  wide margin. If 1.7B is competent and the probe shows headroom, 4B is the follow-up.
- **Base vs post-trained:** use the post-trained `1.7b`, the same choice as `gemma-2-9b-it`. The probe
  checks `1.7b-base` on the same 200 steps if time allows.
- **Licence:** Apache-2.0 needs no click-through. Weights come from Kaggle Models, attached to the
  kernel, so no Hugging Face access is needed.

## The real risk: fp16 on a T4

The T4 has no bf16. Qwen's docs advise against fp16, and there are reports of fp16 NaNs in other
Qwen models (Qwen3-Embedding, Qwen2-VL). Nothing is published about Qwen3 LoRA training in fp16 on a
T4. Mitigations, built in from the start:

- The base model runs in fp16. LoRA parameters and the classification head stay in **fp32**.
- autocast(fp16) plus a GradScaler with gradient clipping. This is the rung 5 setup, *minus* rung 5's
  mistake: weights are cast explicitly and every dtype is checked.
- The final norm and the score head run in fp32.
- Attention is sdpa or eager (FlashAttention 2 does not support Turing). The probe checks that their
  logits agree.
- Any non-finite loss stops the run and names the first layer that overflowed (a forward-hook probe).
- **Fallback:** base weights in fp32. 1.7B is about 8.2 GB, which fits, but expect it to run several
  times slower.

## Fixed inputs (same as every rung)

Everything is reused from rungs 5 and 6, nothing copied:

- **Split:** the same split, with digest and `train.csv` hash checks in the kernel.
- **Truncation and inputs:** `truncate_turns` with per-row adaptive allocation at T = 2,048, the same
  template, and a token-exact a/b swap (`lora_inputs`).
- **Training:** a random swap per example (p = 0.5) and selection on an inner holdout of whole groups.
- **Inference:** both orders, averaged.
- **Outputs and scoring:** text-free out-of-fold files, `merge_oof`, `score_oof`.
- **Data:** Kaggle's own competition `train.csv` (byte-identical to ours). No upload, and no internet
  needed for data.

## Training

| Choice | Plan |
|---|---|
| Head | `AutoModelForSequenceClassification` (Qwen3), 3 classes, cross-entropy |
| LoRA | r = 16, alpha = 32, dropout 0.05, all linear layers; head fully trained (`score`) |
| Optimiser | AdamW, LR 1e-4, 5% warmup, cosine decay, 1 epoch, effective batch 16 |
| Memory | Gradient checkpointing, micro-batch from the probe (1-4), dynamic padding, length-grouped batches |
| GPUs | Probe on one T4. Fold 0 uses DDP across both T4s if the probe shows it pays, else one T4 |

## Steps and gates (all free; Kaggle GPU-hours are the budget)

1. **Probe**, at most 1 GPU-hour, one session. It checks:
   - 200 fp16 steps at T = 2,048: finite, with no first-overflow layer;
   - throughput and peak memory, and DDP against a single GPU;
   - sdpa against eager logits;
   - 1,000 rows of inference.

   **G0:** finite loss, and projected fold-0 training plus inference at most 10 wall-hours, else stop
   and re-plan.
2. **Fold 0:** train on fold 0's training rows, select on its inner holdout, predict fold 0 in both
   orders. Then the **G1** bar, pre-registered:

   | Fold-0 log loss (raw, swap-averaged) | Reading | Next |
   |---|---|---|
   | ≤ 1.00 | Competent (fold 0: LightGBM 1.0425, Haiku 0.9739, DeBERTa 1.0736) | Train the all-rows submission model; plan 4B and the full 5-fold audit run |
   | 1.00 to 1.0425 | Useful only stacked | Check the LightGBM + Qwen stack on fold 0; submit the stack only if it beats LightGBM |
   | > 1.0425 | Not useful | Stop; report it; LightGBM stays the submission |

3. **Submission model** (only if G1 passes): one epoch on all cv rows, an estimated 4-10 h on one T4.
   - The adapter is saved as a private Kaggle dataset, which needs Charles's go.
   - The existing rung 6 submission notebook runs it in fp16 (no 8-bit needed for 1.7B), both orders,
     one model per T4.
   - It needs a timing run first. Then the submission, which also needs Charles's go.

**Budget:** at most 20 Kaggle GPU-hours this week (out of about 30) for the probe, fold 0 and the
submission model. That leaves room for the submission run. Cash cost is $0.

## What this cannot do

- **5-fold out-of-fold predictions:** fold 0 alone gives the G1 read and a fold-0 verbosity readout.
  The full audit comparison needs all five folds, about 5 × 5-10 GPU-hours. That is a follow-up if G1
  passes, spread over about two weeks of quota.
- **A Gemma-class score:** published 9B solutions reach about 0.88-0.90. A 1.7B model is not
  expected to match them; the question is whether it beats LightGBM clearly.

## Work breakdown

1. Generalise `lora_train.py` to take the model name, precision (`bf16` | `fp16-autocast` | `fp32`),
   the fp32 head and norm option, and DDP, keeping the Gemma/Modal path unchanged. Test-first; CPU dry
   run on a tiny Qwen3 config.
2. A kernel push for Kaggle, reusing rung 5's `push.py` pattern: one kernel per session pinned to a
   pushed commit, `model_sources` set to the Qwen3 handle, internet on for the pip install only.
3. The probe, then fold 0, then the G1 decision, scored with `score_oof` restricted to fold 0, or with
   `compare_ablation`'s fold-0 path.
4. If G1 passes: the submission model, the dataset, the timing run and the submission, each needing
   Charles's go.
