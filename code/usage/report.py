"""Write the final-run usage_report.md from one isolated UsageTracker."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from usage.pricing import prices_for_model
from usage.tracker import UsageRecord, UsageTracker

USAGE_REPORT_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "usage_report.md"


def _group(records: list[UsageRecord]) -> dict[tuple[str, str], list[UsageRecord]]:
    grouped: dict[tuple[str, str], list[UsageRecord]] = defaultdict(list)
    for item in records:
        grouped[(item.provider, item.model)].append(item)
    return dict(grouped)


def render_usage_report(
    tracker: UsageTracker,
    *,
    evaluation_requests: int,
    run_id: str,
    timestamp: str | None = None,
    zero_paid_requests: int,
    cached_image_requests: int,
    deterministic_message_requests: int,
    new_model_call_requests: int,
) -> str:
    when = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = tracker.records
    paid = [item for item in records if item.success and not item.cache_hit]
    cache_hits = [item for item in records if item.cache_hit]
    failures = [item for item in records if not item.success]
    total_in = sum(item.input_tokens for item in paid)
    total_out = sum(item.output_tokens for item in paid)
    total_tokens = total_in + total_out
    total_cost = sum((Decimal(item.estimated_cost_usd) for item in paid), Decimal("0"))
    avg_tokens = (Decimal(total_tokens) / evaluation_requests).quantize(Decimal("0.01"))
    avg_cost = (total_cost / evaluation_requests).quantize(Decimal("0.000001"))
    purposes = sorted({item.purpose for item in records})
    models = _group(records)

    lines = [
        "# Final-run usage report",
        "",
        "This report corresponds **only** to the final output-producing run that wrote",
        "the repository-root `output.csv`. Development experiments in",
        "`evaluation/usage.jsonl` are not included.",
        "",
        "## Run summary",
        "",
        f"- total evaluation requests: {evaluation_requests}",
        f"- run timestamp (UTC): {when}",
        f"- run id: `{run_id}`",
        f"- AI purposes recorded: {', '.join(purposes) if purposes else 'none'}",
        "",
        "## Pricing",
        "",
        "Estimated cost uses the configured table in `code/usage/pricing.py`",
        "(OpenAI published list prices baked in at implementation time, overridable via",
        "`BUYORWAIT_PRICE_INPUT_PER_TOKEN` / `BUYORWAIT_PRICE_OUTPUT_PER_TOKEN`).",
        "This environment did not perform a live official-pricing lookup for the final run.",
        "",
    ]
    seen_models = sorted({item.model for item in records} | {"gpt-4o-mini"})
    for model in seen_models:
        inp, out = prices_for_model(model)
        lines.append(
            f"- `{model}`: input {format(inp, 'f')} USD/token, output {format(out, 'f')} USD/token"
        )
    lines.extend(["", "## Per-model", ""])
    if not models:
        lines.append("No model-access records were written during the final run.")
        lines.append("")
    for (provider, model), items in sorted(models.items()):
        model_paid = [item for item in items if item.success and not item.cache_hit]
        model_hits = [item for item in items if item.cache_hit]
        model_fail = [item for item in items if not item.success]
        inp = sum(item.input_tokens for item in model_paid)
        out = sum(item.output_tokens for item in model_paid)
        cost = sum((Decimal(item.estimated_cost_usd) for item in model_paid), Decimal("0"))
        lines.extend(
            [
                f"### {provider} / {model}",
                "",
                f"- paid calls: {len(model_paid)}",
                f"- cache hits: {len(model_hits)}",
                f"- failed calls / retries recorded: {len(model_fail)}",
                f"- input tokens (paid only): {inp}",
                f"- output tokens (paid only): {out}",
                f"- total tokens (paid only): {inp + out}",
                f"- estimated cost (USD): {cost}",
                "",
            ]
        )
    lines.extend(
        [
            "## Overall",
            "",
            f"- total recorded accesses (paid + cache + failures): {len(records)}",
            f"- paid calls: {len(paid)}",
            f"- cache hits: {len(cache_hits)}",
            f"- failed calls: {len(failures)}",
            f"- total paid tokens: {total_tokens}",
            f"- average paid tokens per evaluation request: {avg_tokens}",
            f"- total estimated cost (USD): {total_cost}",
            f"- average estimated cost per request (USD): {avg_cost}",
            "",
            "## Selective-AI summary",
            "",
            f"- requests with zero paid model calls during the final run: {zero_paid_requests}",
            f"- requests that used cached image evidence: {cached_image_requests}",
            f"- requests resolved with deterministic message parsing only: {deterministic_message_requests}",
            f"- requests that triggered a new paid model call: {new_model_call_requests}",
            "",
        ]
    )
    if not paid:
        lines.extend(
            [
                "The final output-producing run made **zero paid model calls**.",
                "All image evidence came from the existing cache; messages were parsed",
                "deterministically. Cache hits are recorded with zero tokens and zero cost.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_usage_report(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
