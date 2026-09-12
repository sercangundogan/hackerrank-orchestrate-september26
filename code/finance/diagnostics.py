"""Concise normalization diagnostics for later sample debugging."""

from __future__ import annotations

from collections import Counter

from decision.models import CapacityResult
from evidence.models import EvidenceBundle
from finance.forecast_models import ForecastResult
from finance.models import NormalizedFinancialState


def _fmt_amount(value) -> str:
    if value is None:
        return "unresolved"
    return f"{value} {''}".rstrip()


def _event_line(event) -> str:
    amount = (
        f"{event.amount_home_currency}"
        if event.amount_home_currency is not None
        else "amount=unresolved"
    )
    cash_date = event.cash_date.isoformat() if event.cash_date else "no-cash-date"
    return (
        f"{event.source_event_id} {event.status.value} {event.direction.value} "
        f"{amount} {event.category} on {cash_date} ({event.description})"
    )


def format_financial_state_diagnostics(
    state: NormalizedFinancialState,
    *,
    max_rows: int = 6,
) -> str:
    ignored_reasons = Counter(
        event.ignore_reason.value for event in state.ignored_events if event.ignore_reason
    )
    lines = [
        f"Financial state for {state.request_id}",
        f"User: {state.user_id}  request_date: {state.request_date.isoformat()}",
        f"Opening balance: {state.opening_balance} {state.profile.home_currency.value}",
        f"Minimum balance: {state.profile.minimum_balance_to_keep} {state.profile.home_currency.value}",
        "",
        f"Historical settled cash evidence: {len(state.historical_settled_cash_events)}",
    ]
    for event in state.historical_settled_cash_events[:max_rows]:
        lines.append(f"  {_event_line(event)}")
    if len(state.historical_settled_cash_events) > max_rows:
        lines.append(
            f"  ... {len(state.historical_settled_cash_events) - max_rows} more"
        )

    lines.append("")
    lines.append(f"Pending debits: {len(state.pending_debits)}")
    for event in state.pending_debits:
        lines.append(f"  {_event_line(event)}")

    lines.append("")
    lines.append(f"Scheduled debits: {len(state.scheduled_debits)}")
    for event in state.scheduled_debits:
        lines.append(f"  {_event_line(event)}")

    lines.append("")
    lines.append(f"Confirmed scheduled income: {len(state.confirmed_scheduled_income)}")
    for event in state.confirmed_scheduled_income:
        lines.append(f"  {_event_line(event)}")

    lines.extend(
        [
            "",
            "Ignored:",
            f"  pending credits: {ignored_reasons.get('pending_credit', 0)}",
            f"  cancelled: {ignored_reasons.get('cancelled', 0)}",
            f"  failed: {ignored_reasons.get('failed', 0)}",
            f"  unrealized: {ignored_reasons.get('unrealized', 0)}",
            f"  other: {sum(ignored_reasons.values()) - sum(ignored_reasons.get(key, 0) for key in ('pending_credit', 'cancelled', 'failed', 'unrealized'))}",
        ]
    )

    lines.append("")
    lines.append(f"Linked lifecycles: {len(state.lifecycle_groups)}")
    for group in state.lifecycle_groups:
        lines.append(
            f"  {group.parent_event_id} -> {group.child_event_id}: "
            f"{group.lifecycle_type.value} ({group.cash_effect_summary})"
        )

    likely = [series for series in state.recurring_series_candidates if series.likely_recurring]
    lines.append("")
    lines.append(
        f"Recurring candidates: {len(likely)} likely / {len(state.recurring_series_candidates)} grouped"
    )
    for series in likely[:12]:
        lines.append(
            f"  {series.inferred_cadence.value}/{series.cadence_confidence.value} "
            f"{series.category} {series.original_description!r} "
            f"n={len(series.event_ids)} {series.amount_behavior.value} "
            f"flex={series.flexibility.value}"
        )

    lines.append("")
    lines.append(f"Unresolved: {len(state.unresolved_events)}")
    for event in state.unresolved_events:
        lines.append(
            f"  {event.source_event_id}: {event.ignore_reason.value if event.ignore_reason else 'unresolved'}"
            + (
                " (blank amount requiring image)"
                if event.requires_external_evidence
                else ""
            )
        )

    if state.requires_message_resolution:
        lines.append("")
        lines.append(
            f"Messages present (not interpreted): {', '.join(state.user_message_ids) or 'none'}"
        )
    if state.user_image_ids:
        lines.append(f"Images present (not read): {', '.join(state.user_image_ids)}")
    if state.direction_inconsistencies:
        lines.append("Direction inconsistencies:")
        lines.extend(f"  {item}" for item in state.direction_inconsistencies)
    if state.flexibility_inconsistencies:
        lines.append("Flexibility inconsistencies:")
        lines.extend(f"  {item}" for item in state.flexibility_inconsistencies)
    return "\n".join(lines)


def format_forecast_diagnostics(result: ForecastResult) -> str:
    """Print only days that have cash events."""
    lines = [
        f"Forecast for {result.request_id}",
        "",
        f"Opening balance: {result.opening_balance}",
        f"Minimum balance: {result.minimum_balance_to_keep}",
        f"Horizon: {result.horizon_start.isoformat()} through {result.horizon_end.isoformat()} (inclusive)",
        "",
    ]
    if not result.daily_forecasts:
        lines.append("No forecast cash events in the horizon.")
        lines.append("")
    for day in result.daily_forecasts:
        lines.append(day.date.isoformat())
        for entry, balance in zip(day.entries, day.running_balances):
            sign = "+" if entry.signed_amount >= 0 else "-"
            lines.append(f"{sign}{entry.amount_home_currency} {entry.description}")
            lines.append(f"balance: {balance}")
        lines.append("")

    lines.append(f"Minimum observed balance: {result.minimum_observed_balance}")
    if result.is_safe:
        lines.append("Minimum violation: no")
    else:
        entry = result.first_violation_entry
        detail = (
            f"{result.first_violation_date.isoformat()} ({entry.description})"
            if result.first_violation_date and entry is not None
            else (
                result.first_violation_date.isoformat()
                if result.first_violation_date
                else "yes"
            )
        )
        lines.append(f"Minimum violation: {detail}")

    lines.append("")
    lines.append("Generated recurrence:")
    if result.generated_recurrence_summaries:
        lines.extend(f"  {item}" for item in result.generated_recurrence_summaries)
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Ignored:")
    if result.ignored_summaries:
        lines.extend(f"  {item}" for item in result.ignored_summaries)
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Unresolved:")
    if result.unresolved_reasons:
        lines.extend(f"  {item}" for item in result.unresolved_reasons)
    else:
        lines.append("  none")

    if result.message_uncertainty_flags:
        lines.append("")
        lines.append("Message uncertainty (not interpreted):")
        lines.extend(f"  {item}" for item in result.message_uncertainty_flags)
    return "\n".join(lines)


def format_evidence_diagnostics(bundle: EvidenceBundle) -> str:
    lines = [
        f"Evidence for {bundle.request_id}",
        f"Facts: {len(bundle.facts)}",
        "",
    ]
    for fact in bundle.facts:
        amount = f" {fact.amount} {fact.currency.value}" if fact.amount is not None and fact.currency else ""
        when = f" on {fact.effective_date.isoformat()}" if fact.effective_date else ""
        lines.append(
            f"{fact.source_id} {fact.fact_type.value}{amount}{when} "
            f"[{fact.extraction_method.value}/{fact.confidence.value}] {fact.notes}"
        )
    if bundle.image_extractions:
        lines.append("")
        lines.append("Images:")
        for item in bundle.image_extractions:
            if item.amount is None:
                lines.append(
                    f"  {item.image_id} unresolved ({item.unresolved_reason or item.rationale})"
                )
            else:
                currency = item.currency.value if item.currency else "?"
                lines.append(
                    f"  {item.image_id} {item.amount} {currency} "
                    f"{item.confidence.value} {item.selected_label or ''} — {item.rationale}"
                )
    if bundle.image_reviews:
        lines.append("")
        lines.append("Image reviews:")
        for reviewed in bundle.image_reviews:
            original = reviewed.original
            status = reviewed.validation.status.value
            accepted = "accepted" if reviewed.accepted else "unresolved"
            lines.append(
                f"  {original.image_id} {status} {accepted} "
                f"field={reviewed.final_field or original.selected_label or '?'} "
                f"amount={reviewed.final_amount if reviewed.accepted else original.amount} "
                f"— {reviewed.validation.validation_reason}"
            )
    if bundle.failures:
        lines.append("")
        lines.append("Failures:")
        lines.extend(f"  {item}" for item in bundle.failures)
    if bundle.llm_source_ids:
        lines.append("")
        lines.append("LLM candidates: " + ", ".join(bundle.llm_source_ids))
    return "\n".join(lines)


def format_capacity_diagnostics(result: CapacityResult) -> str:
    lines = [
        f"Capacity for {result.request_id}",
        "",
        f"Requested amount: {result.requested_amount}",
        f"Safe today: {result.amount_safe_to_pay}",
        f"Full safe today: {'yes' if result.full_payment_safe_today else 'no'}",
        (
            f"Earliest full payment: {result.earliest_date_for_full_payment.isoformat()}"
            if result.earliest_date_for_full_payment is not None
            else "Earliest full payment: none"
        ),
        "",
        "Limiting baseline point:",
    ]
    if result.limiting_date is not None:
        lines.append(result.limiting_date.isoformat())
        if result.limiting_event is not None:
            lines.append(f"Event: {result.limiting_event.description}")
        lines.append(f"Minimum required: {result.baseline.minimum_balance_to_keep}")
    else:
        lines.append("none")
    lines.append(f"Projected baseline minimum: {result.baseline_minimum_balance}")
    lines.append("")
    lines.append(
        f"Payment of {result.amount_safe_to_pay}: "
        f"minimum observed = {result.safe_payment_forecast_minimum} → "
        f"{'SAFE' if result.safe_today_forecast.is_safe else 'UNSAFE'}"
    )
    if result.unsafe_increment_minimum is not None:
        bump = result.amount_safe_to_pay + result.quantum
        lines.append(
            f"Payment of {bump}: "
            f"minimum observed = {result.unsafe_increment_minimum} → UNSAFE"
        )
    if result.unresolved_reasons:
        lines.append("")
        lines.append("Unresolved:")
        lines.extend(f"  {item}" for item in result.unresolved_reasons)
    return "\n".join(lines)
