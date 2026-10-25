# Rung 6: Gemma-2-9B LoRA on Modal

Plan and gates: `docs/plans/rung-6-lora.md` (on `docs/rung-6-plan`). Code:

| File | Role |
|---|---|
| `llm_preference_audit/lora_train.py` | training and inference (probe / fold / full / selftest); CPU `--tiny` dry runs |
| `llm_preference_audit/lora_inputs.py`, `lora_cost.py`, `kernel_common.py` | torch-free: template assembly, cost maths, data-integrity constants |
| `lora/modal_app.py` | the Modal app and local entrypoint |
| `lora/spend_guard.py`, `lora/launch_plan.py` | spend guard (G0, approved amount, $85 ceiling, G1) and preflight |
| `lora/train_lora.py` | CLI wrapper around `lora_train` |

One-time setup (yours, not scripted): `modal token new`; `modal secret create huggingface HF_TOKEN=...`
after accepting the Gemma terms on Hugging Face. The first `modal run` creates the volume
`llm-pref-audit`. The Modal image is built from the pinned commit, so commit and push first (the
entrypoint refuses a dirty tree or an unpushed SHA).

```sh
export LORA_SHA=$(git rev-parse HEAD)
```

## CPU dry runs (no Modal, no spend)

```sh
uv run --with torch --with transformers --with peft --with sentencepiece --with protobuf \
  python lora/train_lora.py --mode selftest --tiny
uv run --with torch --with transformers --with peft --with sentencepiece --with protobuf \
  python lora/train_lora.py --mode probe --tiny --max-rows 300 --t 256 --probe-steps 5 \
  --probe-infer-rows 20 --attn-rows 40 --data-dir data/raw --out-dir /tmp/lora-dry/probe
uv run --with torch --with transformers --with peft --with sentencepiece --with protobuf \
  python lora/train_lora.py --mode fold --fold 0 --tiny --max-rows 300 --t 256 \
  --data-dir data/raw --out-dir /tmp/lora-dry/fold
```

The tiny path downloads the open Qwen2.5 tokenizer (Gemma's is gated). Add `--dry` to
`modal run` to see the spend-guard verdict without uploading or launching anything.

## Prefetch, deadlines, and what the guard enforces

* **Weights are prefetched on CPU.** Before the first launch the entrypoint runs `prefetch()` (no GPU,
  needs the `huggingface` secret) if `/vol/hf-cache/.prefetched/gemma-2-9b-it` is absent, and records
  it in the spend file as CPU spend at $0.20/h x the 30 min timeout = $0.10 (Modal lists $0.0000131
  per core-s and $0.00000222 per GiB-s; 2 cores + 8 GiB is about $0.16/h, modal.com/pricing,
  2026-10-25). GPU functions run with `HF_HUB_OFFLINE=1` and `--local-files-only` and never see
  `HF_TOKEN`; a missing cache fails before any model load.
* **Time limits come from the budget.** `deadline_s = min(6 h, remaining_approved x 3600 / 3.95 /
  n_jobs) - 10 min`, where remaining_approved is the approved amount (or $85, whichever binds) minus
  recorded spend; a probe is also capped at $5. The launch is refused if that is shorter than the
  job's need (probe projection x 1.15; 20 min without a projection). The same number is passed as the
  Modal timeout (`Function.with_options(timeout=deadline+10 min)`, per launch) and as `--deadline-s`
  (an in-process wall clock: training and inference stop cleanly, the log gets `error="deadline"`,
  `probe.json` is rewritten after start, setup, the attention check, training and inference).
  The worst case recorded at launch is timeout x price x jobs.
* **Spend is charged at the worst case until reconciled** (max of estimate and worst case). The
  automatic reconcile charges each job its log's `est_cost_usd` x 1.15 (at most its worst case), or its
  full worst case when it left no log, and never less than min(estimate, worst case). Only
  `reconcile --job-id J --actual-usd X --source modal-dashboard` can lower it; it records the source
  and the previous value, and the spend file is append-only. Folds 1-4 launched together are charged
  their whole remaining-budget worst case, so `full` waits until they are reconciled.
* **One launch at a time.** `data/lora_out/launch.lock` (O_CREAT|O_EXCL) is held from the guard check
  through `record_launch`. A leftover lock refuses; read it and remove it by hand.
* **A tag is single-use locally**: the launch refuses if `data/lora_out/<tag>/` already holds this
  launch's log or probe file (the automatic reconcile would read an old run's cost). In the container
  each run deletes its own stale `oof_fold{k}.parquet`/log (and a full run its adapter files) first.
* **Preemption.** Functions use `retries=0`. Modal's preemption guide says a preempted input is
  restarted on the same input and does not say whether that counts against retries;
  `nonpreemptible` is not supported for GPU functions. A preemption can therefore re-bill an input
  beyond the one-attempt worst case: check the dashboard after every launch.
* `--mode fetch` needs no `LORA_SHA` (the GPU functions and image only exist when it is set).

## G0: probe (at most $5; one-hour function timeout = $3.95)

```sh
modal run lora/modal_app.py --mode probe --tag probe --sha $LORA_SHA --yes
```

Writes `data/lora_out/probe/probe.json`: throughput, peak memory, the eager-vs-sdpa decision, and the
cost projection (fold, 5 folds, all-cv adapter, plus the 15% overhead line). **G0 is machine-enforced**:
any fold or full launch needs a complete probe for the SAME git sha (not tiny, google/gemma-2-9b-it,
T=2048, loss finite, `non_finite_loss` null, an attention decision, and
`projection.total_usd_with_overhead` <= $70). The probe is chosen by sha and realness, never by mtime
or tag, and drives `--attn auto`. Any code change needs a new probe.
If memory is tight, lower `--micro-batch` (must divide 16; throughput and projection change).

## G1: fold 0 (one container)

```sh
modal run lora/modal_app.py --mode fold --folds 0 --tag g1 --sha $LORA_SHA --yes
uv run python lora/spend_guard.py g1 --oof data/lora_out/g1/oof_fold0.parquet \
  --probe data/lora_out/probe/probe.json        # writes data/lora_out/g1_decision.json
```

`pass` means swap-averaged raw fold-0 log loss <= 1.00 (1.0001 fails). The decision records the
fold-0 log's git sha and tag, the OOF file's sha256 and the verified train.csv sha256 (the CLI
refuses a raw file whose hash is not `TRAIN_SHA256`, and a fold log that errored). Folds 1-4 and `full`
re-score that exact OOF file at launch (hash must match, loss recomputed) and need the launch's sha to
equal the decision's. Between 1.00 and 1.0425
stop and report; above 1.0425 debug one round (`--lr` is not a CLI flag of the launcher: change
`LR` in `lora_train.py`, commit, push, rebuild).

## Folds 1-4 and the submission adapter (auto-approved only after G1 passes)

```sh
modal run lora/modal_app.py --mode fold --folds 1,2,3,4 --tag f1234 --sha $LORA_SHA --yes
modal run lora/modal_app.py --mode full --tag full --sha $LORA_SHA --yes
```

Folds run concurrently, one H100 container each (`max_containers=5`). The guard allows them up to
the probe projection x 1.2 on top of the approved $15, and never past $85 cumulative. Raise the
approval explicitly with `--approved-usd 30` if you decide to skip the G1 projection. The adapter
lands on the volume at `/vol/adapters/full/` (download it with the Modal dashboard or
`modal volume get llm-pref-audit adapters/full ...`); it is private and never goes in git.

## Fetch, merge, score, record

Each launch downloads its outputs automatically; re-fetch with `--mode fetch --tag <tag>`.

```sh
uv run python kaggle/rung5_deberta/merge_oof.py \
  --in data/lora_out/g1 --in data/lora_out/f1234 --tag lora_gemma2_9b_t2048
cat data/predictions/lora_gemma2_9b_t2048.run.json    # gpu_hours_estimate, est_cost_usd
uv run python scripts/score_oof.py --oof data/predictions/lora_gemma2_9b_t2048.parquet \
  --name lora_gemma2_9b_t2048 --model google/gemma-2-9b-it \
  --gpu-hours <gpu_hours_estimate> --cost-usd <real Modal bill> --ledger
```

`merge_oof.py` is architecture-agnostic: it validates every fold's ids against the manifest and
that all logs share `git_sha`, `t`, `middle_share`, `model`, `epochs`. Keep the ledger `--name`
starting with `lora` so the guard counts the row as rung-6 spend (it takes the larger of the spend
file and the ledger rung-6 total, never both).

## Cost record

* Every launch appends its estimate to `data/lora_out/spend.jsonl` BEFORE launching; afterwards the
  entrypoint appends a reconcile line (`est_cost_usd` in the logs x 1.15 for container boot).
  Replace it with the Modal dashboard figure: `python lora/spend_guard.py reconcile --job-id <id>
  --actual-usd <dollars> --source modal-dashboard`. `python lora/spend_guard.py status` prints totals.
* `est_cost_usd` in each log = wall seconds x $3.95/3600 (H100 list price, modal.com/pricing,
  2026-10-25). Container boot and image pull are not in it.
* A stuck job is bounded by its budget-derived Modal timeout (see above), not a fixed 6 h. Check
  `modal app list` if a launch dies mid-way.
* Per-fold config is verified against the launcher's request inside the container
  (`check_worker_config`). Folds re-encode the 51,848 cv rows (a few CPU minutes, billed).
