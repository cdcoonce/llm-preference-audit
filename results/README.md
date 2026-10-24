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

Rung 2 rows also carry `config`, `predictions_sha256`, `n_rows`, `ci95_vs_prior` and
`best_iterations`. Out-of-fold prediction files live in `data/predictions/` (gitignored) and are
identified by the sha256 in the row.

Ledger rows are never edited. A later row with the same config and identical scores supersedes an
earlier one: the first three rung 2 rows (git_sha edc4806) are superseded by the d30e8f1 rows.

`rung-2-summary.json` holds the aggregates behind `docs/rung-2-features.md`.
