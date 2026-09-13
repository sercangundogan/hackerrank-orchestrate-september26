# Final-run usage report

This report corresponds **only** to the final output-producing run that wrote
the repository-root `output.csv`. Development experiments in
`evaluation/usage.jsonl` are not included.

## Run summary

- total evaluation requests: 250
- run timestamp (UTC): 2026-09-12T23:22:03Z
- run id: `final-output-20260912T232200Z`
- AI purposes recorded: image_amount_extract

## Pricing

Estimated cost uses the configured table in `code/usage/pricing.py`
(OpenAI published list prices baked in at implementation time, overridable via
`BUYORWAIT_PRICE_INPUT_PER_TOKEN` / `BUYORWAIT_PRICE_OUTPUT_PER_TOKEN`).
This environment did not perform a live official-pricing lookup for the final run.

- `gpt-4o-mini`: input 0.00000015 USD/token, output 0.00000060 USD/token

## Per-model

### cache / gpt-4o-mini

- paid calls: 0
- cache hits: 11
- failed calls / retries recorded: 0
- input tokens (paid only): 0
- output tokens (paid only): 0
- total tokens (paid only): 0
- estimated cost (USD): 0

## Overall

- total recorded accesses (paid + cache + failures): 11
- paid calls: 0
- cache hits: 11
- failed calls: 0
- total paid tokens: 0
- average paid tokens per evaluation request: 0.00
- total estimated cost (USD): 0
- average estimated cost per request (USD): 0.000000

## Selective-AI summary

- requests with zero paid model calls during the final run: 250
- image accesses during the final run: 11 cache hits, 0 paid vision calls
- evaluation requests that required a paid text-model message parse: 0
- requests that triggered a new paid model call: 0

The final output-producing run made **zero paid model calls**.
The 11 recorded image accesses were cache hits against the existing validated
evidence cache; no paid vision calls were made. Message evidence that appeared
for a request was parsed deterministically. That does not mean every evaluation
request contained a message. Cache hits are recorded with zero tokens and zero
cost.

