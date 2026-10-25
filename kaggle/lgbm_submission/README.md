# Rung 2 LightGBM submission kernel (CPU, internet off, no dataset)

A code-competition entry for `llm-classification-finetuning` from rung 2's `full` LightGBM config.
The kernel is ONE generated Python script. `build_kernel.py` inlines the repo modules it needs
(`data, split, kernel_common, profile, truncate, cv_common, deberta_inputs, features, submission,
train_lgbm`) from a pinned clean commit into a namespace package registered in `sys.modules`, so
the code that runs is exactly the committed code. Third-party imports are only numpy, pandas and
lightgbm, all in Kaggle's image (a test checks the import closure). Nothing is attached.

| File | Role |
|---|---|
| `build_kernel.py` | preflight (HEAD pinned, `llm_preference_audit/` and `kaggle/` clean, HEAD on a remote branch; reuses rung 5's `preflight`), renders `data/kaggle_build/lgbm-sub/`, PRINTS the push/submit commands |
| `kernel_main.py` | the kernel's main, appended to the inlined modules (also importable by pytest) |
| `dry_run_input.py` | builds a fake `/kaggle/input` tree for a local dry run |

What the kernel does: writes a prior-filled `submission.csv` at once; verifies `train.csv`
(sha256 = `TRAIN_SHA256`) and the split manifest digest; featurizes and fits ONLY `split == "cv"`
rows (the locked test split is never fitted on; `assert_cv_only` backs it); fits one LightGBM with
rung 2's `PARAMS` and swap augmentation for a fixed number of rounds (no early-stopping holdout);
predicts the competition `test.csv` swap-averaged; overwrites `submission.csv` atomically
(`id, winner_model_a, winner_model_b, winner_tie`, test-file order) and writes `run_log.json`
(row counts, rounds, seconds per phase, versions, source commit, module hashes; no text).

Rounds = `round(mean(best_iterations of the ledger's latest lightgbm full row) x 1.1)`, computed by
the build and baked into the script header. The final model sees ~1.4x the rows of a CV fold's fit
set, hence the 1.1 margin.

## 1. Build

```
uv run python kaggle/lgbm_submission/build_kernel.py            # commit + push first
```

Refuses a dirty tree, a `--sha` that is not HEAD, or a HEAD not on a remote branch. `--dev --out DIR`
skips the preflight and reads the working tree (dry runs only; the header says `commit dev`).

## 2. Local dry run (no network, no Kaggle)

```
S=<scratch dir outside the repo>
uv run python kaggle/lgbm_submission/build_kernel.py --dev --out $S/render
uv run python kaggle/lgbm_submission/dry_run_input.py --source cv-fold0 --out $S/fake
LPA_INPUT_ROOT=$S/fake/input LPA_OUTPUT_DIR=$S/fake/work \
  .venv/bin/python -I $S/render/lgbm_submission.py
```

`--source test` uses 500 rows of the locked test split: it checks the pipeline runs and must NOT be
scored. `--source cv-fold0` is in-sample for the final model (a sanity check, expect a log loss
below rung 2's OOF fold-0 1.0425, not a score).

## 3. Push (needs Charles's go)

```
kaggle kernels push -p data/kaggle_build/lgbm-sub
kaggle kernels status charlescoonce/llm-pref-lgbm-sub
```

Metadata: private, `kernel_type` script, `enable_gpu` false, `enable_internet` false,
`competition_sources` = the competition, no dataset/model/kernel sources.

## 4. Check the kernel output

```
kaggle kernels output charlescoonce/llm-pref-lgbm-sub -p data/kaggle_out/lgbm-sub
```

Read `run_log.json` (`n_cv_rows` 51848, `rounds`, seconds, `source_commit` = the pushed SHA,
`skip_sha_checks` false) and confirm `submission.csv` has the competition's visible `test.csv`
rows with finite probabilities summing to 1. On the visible 3-row sample the file is tiny; real
scoring happens on the hidden test.

## 5. Submit (needs Charles's go)

```
kaggle competitions submit -k charlescoonce/llm-pref-lgbm-sub -v <VERSION> -f submission.csv -m "rung 2 LightGBM full"
```

`-k` + `-v` pick the kernel version, `-f` names the output file it produced. In a code competition
this does not upload a file: Kaggle RE-RUNS that kernel version on the hidden test set (internet off,
the real `test.csv` mounted in place of the sample), takes its `submission.csv`, and scores it. So
the kernel must finish within the competition's runtime limit on the full hidden test and must not
depend on anything outside `/kaggle/input`.
