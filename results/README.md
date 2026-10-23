# results

Planned: an append-only `ledger.jsonl`, one JSON row per run, never rewritten. Fields:

- `run_id`
- `timestamp_utc`
- `git_sha`
- `config_hash`
- `model` (or `judge`)
- `split`
- `fold_scores`
- `log_loss`
- `cost_usd`
- `predictions_path`

The file is empty until the first run.
