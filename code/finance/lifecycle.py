"""Classify linked-event pairs without collapsing them into one cash row."""

from __future__ import annotations

from data.models import EventStatus, EventType, FinancialEvent
from finance.models import EventLifecycle, LifecycleType


def classify_lifecycle(parent: FinancialEvent, child: FinancialEvent) -> EventLifecycle:
    """Classify one parent→child link from types, statuses, and descriptions."""
    lifecycle_type = _lifecycle_type(parent, child)
    return EventLifecycle(
        parent_event_id=parent.event_id,
        child_event_id=child.event_id,
        lifecycle_type=lifecycle_type,
        cash_effect_summary=_cash_effect_summary(lifecycle_type),
    )


def _lifecycle_type(parent: FinancialEvent, child: FinancialEvent) -> LifecycleType:
    parent_key = (parent.event_type, parent.status)
    child_key = (child.event_type, child.status)

    if parent_key == (EventType.EXPENSE, EventStatus.SETTLED) and child_key == (
        EventType.REFUND,
        EventStatus.SETTLED,
    ):
        return LifecycleType.CHARGE_THEN_REFUND
    if parent_key == (EventType.EXPENSE, EventStatus.SETTLED) and child_key == (
        EventType.REFUND,
        EventStatus.PENDING,
    ):
        return LifecycleType.PURCHASE_THEN_PENDING_REFUND
    if parent_key == (EventType.EXPENSE, EventStatus.CANCELLED) and child_key == (
        EventType.EXPENSE,
        EventStatus.SETTLED,
    ):
        return LifecycleType.AUTHORIZATION_THEN_CAPTURE
    if parent_key == (EventType.DEBT_PAYMENT, EventStatus.FAILED) and child_key == (
        EventType.DEBT_PAYMENT,
        EventStatus.SCHEDULED,
    ):
        return LifecycleType.FAILED_THEN_RETRY
    if parent_key == (
        EventType.INVESTMENT_PURCHASE,
        EventStatus.SETTLED,
    ) and child_key == (EventType.INVESTMENT_VALUATION, EventStatus.UNREALIZED):
        return LifecycleType.INVESTMENT_PURCHASE_THEN_VALUATION
    if parent_key == (
        EventType.INVESTMENT_PURCHASE,
        EventStatus.SETTLED,
    ) and child_key == (EventType.INVESTMENT_SALE, EventStatus.SETTLED):
        return LifecycleType.INVESTMENT_PURCHASE_THEN_SALE
    if parent_key == (EventType.EXPENSE, EventStatus.SETTLED) and child_key == (
        EventType.EXPENSE,
        EventStatus.PENDING,
    ):
        if _looks_like_possible_duplicate(parent, child):
            return LifecycleType.POSSIBLE_DUPLICATE_PENDING_CHARGE
        return LifecycleType.UNKNOWN
    return LifecycleType.UNKNOWN


def _looks_like_possible_duplicate(parent: FinancialEvent, child: FinancialEvent) -> bool:
    child_text = f"{child.description} {child.category}".lower()
    parent_text = parent.description.lower()
    return "duplicate" in child_text or (
        "original" in parent_text and "duplicate" in child_text
    )


def _cash_effect_summary(lifecycle_type: LifecycleType) -> str:
    return {
        LifecycleType.CHARGE_THEN_REFUND: (
            "both rows may be cash: settled debit then settled refund credit"
        ),
        LifecycleType.AUTHORIZATION_THEN_CAPTURE: (
            "cancelled parent is not cash; settled child is the purchase"
        ),
        LifecycleType.PURCHASE_THEN_PENDING_REFUND: (
            "settled purchase is historical cash; pending refund is unavailable"
        ),
        LifecycleType.FAILED_THEN_RETRY: (
            "failed parent is not cash; scheduled retry is a prospective obligation"
        ),
        LifecycleType.INVESTMENT_PURCHASE_THEN_VALUATION: (
            "settled purchase may be historical cash; valuation is non-cash"
        ),
        LifecycleType.INVESTMENT_PURCHASE_THEN_SALE: (
            "both rows may be cash at their own settlement dates"
        ),
        LifecycleType.POSSIBLE_DUPLICATE_PENDING_CHARGE: (
            "settled original is historical cash; pending duplicate remains classified "
            "as a prospective debit and is not auto-discarded"
        ),
        LifecycleType.UNKNOWN: (
            "linked pair is retained but not classified; do not guess cash effects"
        ),
    }[lifecycle_type]


def build_lifecycles(
    events: tuple[FinancialEvent, ...],
    events_by_id: dict[str, FinancialEvent],
) -> tuple[EventLifecycle, ...]:
    groups: list[EventLifecycle] = []
    for event in events:
        if event.linked_event_id is None:
            continue
        parent = events_by_id[event.linked_event_id]
        groups.append(classify_lifecycle(parent, event))
    return tuple(groups)
