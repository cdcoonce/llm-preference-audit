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

## G0: probe (at most $5; one-hour function timeout = $3.95)

```sh
modal run lora/modal_app.py --mode probe --tag probe --sha $LORA_SHA --yes
```

Writes `data/lora_out/probe/probe.json`: throughput, peak memory, the eager-vs-sdpa decision, and the
cost projection (fold, 5 folds, all-cv adapter, plus the 15% overhead line). G0 passes when the loss is
finite, memory fits, the attention check is decided, and `projection.total_usd_with_overhead` is at
most $70. Later launches read the newest `data/lora_out/*/probe.json` for `--attn auto`.
If memory is tight, lower `--micro-batch` (must divide 16; throughput and projection change).

## G1: fold 0 (one container)

```sh
modal run lora/modal_app.py --mode fold --folds 0 --tag g1 --sha $LORA_SHA --yes
uv run python lora/spend_guard.py g1 --oof data/lora_out/g1/oof_fold0.parquet \
  --probe data/lora_out/probe/probe.json        # writes data/lora_out/g1_decision.json
```

`pass` means swap-averaged raw fold-0 log loss <= 1.00 (1.0001 fails). Between 1.00 and 1.0425
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
  --actual-usd <dollars>`. `python lora/spend_guard.py status` prints totals.
* `est_cost_usd` in each log = wall seconds x $3.95/3600 (H100 list price, modal.com/pricing,
  2026-10-25). Container boot and image pull are not in it.
* A stuck fold is bounded only by its function timeout (6 h = $23.70; override with
  `LORA_TIMEOUT_S=7200`). Check `modal app list` if a launch dies mid-way.
* Per-fold config is verified against the launcher's request inside the container
  (`check_worker_config`). Folds re-encode the 51,848 cv rows (a few CPU minutes, billed).
