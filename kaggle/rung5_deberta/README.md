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
3. Folds, in sessions. One 12 h kernel cannot hold five folds, so each push runs a subset with its
   own tag; the tag also names the kernel (`llm-pref-rung5-deberta-<tag>`) and the output dir, so
   sessions can run concurrently if quota allows and never overwrite each other:
   ```
   push.py --username <you> --mode folds --folds 0,1 --tag f01
   push.py --username <you> --mode folds --folds 2,3 --tag f23
   push.py --username <you> --mode folds --folds 4   --tag f4
   push.py --username <you> --mode folds --folds 0 --middle-share 0 --tag abl0   # ablation
   ```
   The ablation (head+tail only, fold 0) is a separate push; its folds never enter the merge.
   Fetch each: `push.py --username <you> --mode folds --tag f01 --fetch` (to `data/kaggle_out/f01/`).
   Each fold writes `oof_fold{k}.parquet` plus `log_fold{k}.json` (tag, git_sha, t, middle_share,
   model, epochs, micro_batch, seed, fold, timings). The OOF file is written only after its log
   loss is computed; an errored fold leaves a log with `error` and no OOF.
4. Merge and score the main run (three `--in` dirs; the ablation dir is not one of them):
   ```
   uv run python kaggle/rung5_deberta/merge_oof.py --in data/kaggle_out/f01 \
       --in data/kaggle_out/f23 --in data/kaggle_out/f4 --tag deberta_base_t1024
   uv run python scripts/score_oof.py --oof data/predictions/deberta_base_t1024.parquet \
       --name deberta_base_t1024 --ledger ...
   ```
   `merge_oof.py` refuses unless every OOF file has an error-free log, fold-k ids equal the
   manifest's fold-k cv ids, all logs share git_sha/t/middle_share/model/epochs, and folds 0-4
   each appear once. It writes `data/predictions/<tag>.parquet` and `<tag>.run.json` (folds, session
   tags, git_sha, summed GPU seconds and a gpu_hours estimate).
5. Ablation comparison. `score_oof.py` needs all cv rows, so the one-fold ablation is compared on
   fold 0 only, against the main run's fold-0 rows:
   ```
   uv run python kaggle/rung5_deberta/compare_ablation.py \
       --main data/predictions/deberta_base_t1024.parquet \
       --ablation data/kaggle_out/abl0/oof_fold0.parquet
   ```
   Prints fold-0 log loss for both, the paired group-bootstrap difference (positive = main
   better) and each run's calibrated verbosity readout.

Sequence: probe, then f01, f23, f4 (and abl0), fetch each, `merge_oof` with three `--in`,
`score_oof --ledger`, `compare_ablation`.

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
