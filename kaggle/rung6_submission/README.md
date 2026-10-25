# Rung 6 submission: Gemma-2-9B-it + LoRA on Kaggle (T4 x2, internet off)

Inference-only notebook for the code competition `llm-classification-finetuning`. The all-cv
adapter is trained on Modal (`lora/README.md`); this folder turns it into `submission.csv`.

| File | Role |
|---|---|
| `llm_preference_audit/submission.py` | torch-free, CI-tested: input discovery, `load_test`, `split_for_gpus`, token-budget batching, unlabelled encoding (same `fit_pair` as training), `submission_frame`, `prior_submission`, `apply_calibrator`, part merge, timing projection |
| `llm_preference_audit/calibrate.py::export_calibrator` | MatrixScaler -> JSON for `calibrator.json` |
| `kaggle/rung6_submission/infer.py` | the kernel script (main process + one worker per GPU) |
| `kaggle/rung6_submission/build_dataset.py` | builds the private dataset (wheel, adapter, manifest). Never uploads |
| `kaggle/rung6_submission/push.py` | renders and pushes the two kernels. `--dry` prints only |

## Sequence

1. **All-cv adapter exists** (`lora/README.md`, `--mode full`): `adapters/full/` on the Modal volume,
   downloaded to e.g. `data/lora_out/full/adapter` (needs `adapter_model.safetensors`,
   `adapter_config.json`, `adapter_meta.json`, which carries the training T and middle share).
2. **Optional calibrator.** Fit `calibrate.MatrixScaler` on cv OOF probabilities, then
   `json.dump(export_calibrator(scaler), open("calibrator.json", "w"))`. Without it the swap-averaged
   raw probabilities are submitted.
3. **Commit and push** the code (the wheel is built from a clean, pushed HEAD), then
   `uv run python kaggle/rung6_submission/build_dataset.py --adapter data/lora_out/full/adapter --tag full [--calibrator calibrator.json] [--include-wheels peft,bitsandbytes,accelerate,sentencepiece]`.
   It refuses a dirty tree or a HEAD not on the remote, runs `uv build --wheel`, copies the adapter,
   writes `MANIFEST.json` (sha256 of every file, git sha, adapter tag) and `dataset-metadata.json`
   (`charlescoonce/llm-pref-rung6-adapter`, private), and PRINTS the upload command.
   `--include-wheels` runs `uvx pip download --only-binary=:all: --no-deps --python-version 3.13`
   for manylinux x86_64 into `wheels/`. Pin versions with `peft==X` if the image's torch needs it.
4. **Charles approves, then uploads** (outward action): the printed
   `kaggle datasets create -p data/kaggle_build/rung6_dataset -r zip` (later versions:
   `kaggle datasets version -p ... -r zip -m "..."`).
5. **Timing kernel** on 1,000 cv rows:
   `uv run python kaggle/rung6_submission/push.py --mode timing --dry` to review, then without
   `--dry`; fetch with `--mode timing --fetch` (to `data/kaggle_out/rung6_timing/`). `timing.json`
   holds throughput, peak memory per GPU, load/encode/infer seconds and the projection to 25,000 rows
   x 2 orders (`fits_budget`, `fits_9h_with_margin`, safety factor 1.15). `run_log.json` has the
   aggregates; `timing_in_sample_log_loss` is a sanity number (the adapter trained on these rows),
   not a score. Timing mode never writes `submission.csv`.
6. **Submission kernel** only if `fits_9h_with_margin` is true:
   `push.py --mode submission` (same preflight: HEAD clean, on the remote, and equal to the dataset
   manifest's git sha). If the projection misses, lower `--t` only knowingly (the adapter was trained
   at its `adapter_meta.json` T); otherwise rely on the single-order fallback below.
7. **Submit.** In a code competition you submit a finished notebook VERSION; Kaggle re-runs that
   version on the hidden test set (~25,000 rows) with internet off and scores its `submission.csv`.
   The version must also finish on the 3-row public `test.csv` (it does: 3 rows, two workers, model
   load dominates). After the version completes:
   `uv run python kaggle/rung6_submission/push.py --mode submission --print-submit-cmd --version N`
   prints `kaggle competitions submit llm-classification-finetuning -k charlescoonce/llm-pref-rung6-submission -v N -f submission.csv -m "..."`
   (flags checked against `kaggle competitions submit --help`, CLI 2.2.4; not executed). Submitting
   is an outward action: it needs Charles's go. Only the submission kernel can be submitted: the
   timing kernel reads train.csv and writes no submission file.

## What the notebook does

* Verifies every dataset file against `MANIFEST.json` (and refuses unlisted extras and `dev` builds),
  then `pip install --no-index --no-deps --force-reinstall <wheel>`. It checks that `sentencepiece`,
  `peft`, `accelerate`, `bitsandbytes` import and installs any missing one from `wheels/`, else fails
  loud naming the missing package.
* Writes a prior-filled `submission.csv` (class prior from train.csv labels, located by sha256)
  BEFORE any GPU work, so a valid file exists if inference dies. Final file is written only after a
  successful merge and fails loud (exit 3, but the file is still written) if more than 2% of rows
  had to be prior-filled or a worker failed.
* Finds the base model by globbing `/kaggle/input` for a `config.json` with `model_type: gemma2`.
  Loads it as `AutoModelForSequenceClassification` in 8-bit (`llm_int8_skip_modules=["score"]`, fp16),
  eager attention, then the adapter + trained score head with `peft` (`local_files_only`).
* Builds inputs with the training code: `fit_pair` to `field_window(T, template)` (T from
  `adapter_meta.json`, default 2048), per-field tokens, `build_lora_ids`, both orders.
* Splits rows over the GPUs by estimated token load (`split_for_gpus`), one worker process per GPU
  (`CUDA_VISIBLE_DEVICES`), each processing its share longest first in token-budget batches
  (`--batch-tokens 4096`, `--max-batch 8`, halves a batch on CUDA OOM). `part{g}.parquet` (ids and
  probabilities) is flushed every 50 batches; a crashed worker is rerun once and resumes from it.
* **Wall-clock budget** (`--limit-s`, default 8h15m, 5 min reserve): before each batch the worker
  projects `2 x remaining padded tokens / measured rate` against the time left (after 3 warm-up
  batches); if both orders would overrun it runs `ab` only for the rest. Rows not reached before the
  deadline get the prior. `run_log.json` records `rows_by_orders_run`, `rows_single_order`,
  `rows_filled_with_prior`, per-worker detail, phase seconds, peak memory, versions, wheel and git sha,
  `calibrated`. No text is written anywhere; row texts for timing mode live in a temp dir that is
  deleted.
* Merge: both orders -> `(ab + swap(ba)) / 2`; one order -> that order; optional `calibrator.json`;
  `submission_frame` clips to [1e-6, 1], renormalises, and raises unless the ids equal the test ids
  exactly (rows in test order).

## CPU dry run (no Kaggle, no GPU)

`infer.py --no-quant --allow-no-bos --skip-install --allow-dev-manifest --gpus 2 --roots <fake input tree>`
against a tiny Gemma2 + tiny adapter and a fake `/kaggle/input` tree (`build_dataset.py --dev`).
`--dry-run-sleep-s` makes the single-order fallback deterministic.

## Unverified until the timing kernel runs

* Whether Kaggle's image already has `peft`, `bitsandbytes`, `accelerate`, `sentencepiece`, and a
  `bitsandbytes` that supports 8-bit on sm75 with the image's torch. The `wheels/` fallback installs
  with `--no-deps`; wheel/torch compatibility is untested.
* Python 3.13.15, torch 2.11, transformers 5.16.1 are from the rung-5 probe; the code uses
  `from_pretrained(dtype=...)`.
* 8-bit fp16 Gemma-2 inference (training was bf16 with fp32 LoRA masters): fp16 overflow, the
  accuracy cost of int8, and `peft` loading `modules_to_save=score` onto a quantised model. Non-finite
  outputs are counted (`nonfinite_rows`) and prior-filled, not hidden. Compare `timing_in_sample_log_loss`
  against the adapter's training-time numbers before trusting a submission.
* Peak memory and throughput on a T4 at T=2048, `--batch-tokens 4096` (OOM halving is the safety net).
  Both workers load the 18.5 GB checkpoint at once; host RAM during that is untested.
* The kernel metadata: `machine_shape: NvidiaTeslaT4` (two Tesla T4 in the rung-5 probe),
  `model_sources: google/gemma-2/transformers/gemma-2-9b-it/2` (the handle and version 2 were
  confirmed read-only with `kaggle models instances get`, 18.5 GB; whether the kernel mounts it as
  `/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2` is unconfirmed, discovery uses the glob),
  and that `-r zip` directories (`adapter/`, `wheels/`) arrive unzipped under the dataset mount.
* That `kaggle competitions submit -k -v` is accepted for this competition and re-runs the version on
  the hidden set as described; that the hidden test length distribution matches train.
* `isPrivate` in `dataset-metadata.json` is not a documented field; `kaggle datasets create` is
  private by default (no `--public`).
