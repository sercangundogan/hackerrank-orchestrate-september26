"""Concise normalization diagnostics for later sample debugging."""

from __future__ import annotations

from collections import Counter

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
