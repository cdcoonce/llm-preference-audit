# Rung 5: DeBERTa-v3-base fine-tune (Kaggle T4 x2)

3-way a/b/tie classifier over the 5 cv folds. Input `[CLS] prompt [SEP] a [SEP] b [SEP]`, T=1024
tokens, per-row adaptive head/middle/tail truncation (`fit_pair`), hard-cut and flagged when a row
still overflows. Cross-entropy, AdamW 2e-5, 2 epochs, effective batch 16, fp16, seed = 1000 + fold,
a/b swapped with p=0.5 per example per epoch, best epoch by inner-holdout log loss (whole groups
from the training folds; the held-out fold never trains or selects). Inference runs both orders.

`train.py` verifies the train.csv sha256 and the split-manifest digest on the full file first, never
touches test.csv, and writes only ids, probabilities, flags and aggregates. Weights stay in memory.

## Steps

1. Commit and push the code (the kernel installs `llm_preference_audit` at a pinned SHA).
2. Probe (one fold, ~200 steps, both-order inference; writes `probe.json` with throughput, peak
   memory and projected GPU-hours):
   `uv run python kaggle/rung5_deberta/push.py --username <you> --mode probe --tag probe1`
   Inspect: `... push.py --username <you> --tag probe1 --fetch` (output in `data/kaggle_out/probe1/`).
   If memory is tight add `--micro-batch 2` or `--grad-ckpt` to the push.
3. Folds: `... push.py --username <you> --mode folds --folds 0,1,2,3,4 --tag full`, then `--fetch`.
   Each fold is a subprocess pinned to its own GPU when two exist. Ablation (head+tail only):
   `--mode folds --folds 0 --middle-share 0 --tag ablate`.
4. Merge and score:
   ```
   uv run python kaggle/rung5_deberta/merge_oof.py --in data/kaggle_out/full --tag deberta_base_t1024
   uv run python scripts/score_oof.py --oof data/predictions/deberta_base_t1024.parquet \
       --name deberta_base_t1024 ...
   ```
   `merge_oof.py` fails unless every cv id appears exactly once and no text column exists.

`push.py` refuses unless the SHA is HEAD, `llm_preference_audit/` and `kaggle/` are clean and the
SHA is on a remote branch. `--dry` prints the `kaggle kernels push` command without running it.
Rendered builds go to `data/kaggle_build/<tag>/` and outputs to `data/kaggle_out/<tag>/` (both
gitignored under `data/`).

## Local dry run (CPU, tiny from-scratch model)

```
uv run --with torch --with transformers --with sentencepiece --with protobuf \
  python kaggle/rung5_deberta/train.py --mode folds --folds 0 --tiny --max-rows 400 \
  --epochs 1 --t 256 --data-dir data/raw --out-dir /tmp/dry
```

torch and transformers are never project dependencies; CI tests (`tests/test_deberta_inputs.py`)
cover the pure logic in `llm_preference_audit/deberta_inputs.py` without them.
