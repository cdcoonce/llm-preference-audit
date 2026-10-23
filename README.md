# llm-preference-audit

> **Work in progress, not yet released.** Nothing here is a finished result.

Do zero-shot LLM judges such as Claude Haiku, and models trained on human labels, share the
biases of human Chatbot Arena voters? This project audits human preference labels from Chatbot
Arena and measures verbosity bias (longer answers win) and position bias (answer A or B wins
for reasons unrelated to quality), then checks whether judges and trained predictors reproduce
or amplify them.

## Planned ladder

1. Class prior baseline.
2. Hand-built features + LightGBM.
3. Haiku judge with A/B swap and calibration.
4. Jev judge.
5. DeBERTa-v3 fine-tune.
6. Gemma / Qwen LoRA, plus a Kaggle offline submission.

## Evaluation

A locked 10% test split plus 5-fold cross-validation on the rest, both grouped by prompt hash so
the same prompt never appears on both sides. The metric is multiclass log loss over
`winner_model_a`, `winner_model_b`, `winner_tie`. Every run appends a row to
`results/ledger.jsonl`.

## Data

The data is the Kaggle "LLM Classification Finetuning" set, also published on Hugging Face as
`lmsys/lmsys-arena-human-preference-55k`, licensed CC BY-NC 4.0. Download it with `make data`
into `data/raw/`. It is never committed: `data/` is gitignored, and `make leak-check` runs in CI
to fail on any tracked data path, oversized data file, or file carrying the prompt/response text
columns. Committed predictions hold only `id` and three probabilities.

## Quickstart

```
make setup
make ci
```

## License

Code is MIT licensed (see LICENSE). The dataset keeps its own CC BY-NC 4.0 license.
