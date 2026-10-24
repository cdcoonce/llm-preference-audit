<!-- haiku_batch_results.jsonl is hand-built in the SDK's MessageBatchIndividualResponse shape (no API key yet).
Replace it with responses recorded on synthetic pairs once the pilot runs. Synthetic content only. -->
# Test fixtures

`haiku_batch_results.jsonl`: one batch result per line, covering ok, thinking-before-text, refusal,
max_tokens, malformed output, errored and expired cases.

`haiku_batch_results_recorded.jsonl` holds six real results from the 2026-10-24 pilot batch
(three rows, both orders), with custom ids and message ids renamed. They carry only Haiku's
four-key JSON answer and usage, no prompt or response text. The hand-built file stays for the
cases a real run did not produce (refusal, truncation, malformed JSON, errors).
