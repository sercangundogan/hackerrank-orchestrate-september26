from __future__ import annotations

from data.models import EventDirection, EventStatus, EventType
from finance.lifecycle import classify_lifecycle
from finance.models import LifecycleType
from tests.factories import make_event


def test_charge_then_settled_refund_does_not_collapse() -> None:
    parent = make_event(
        event_id="p",
        event_type=EventType.EXPENSE,
        status=EventStatus.SETTLED,
        direction=EventDirection.DEBIT,
        description="Card charge later reversed",
    )
    child = make_event(
        event_id="c",
        event_type=EventType.REFUND,
        status=EventStatus.SETTLED,
        direction=EventDirection.CREDIT,
        linked_event_id="p",
        description="Settled card charge reversal",
    )
    group = classify_lifecycle(parent, child)
    assert group.lifecycle_type is LifecycleType.CHARGE_THEN_REFUND
    assert "both rows may be cash" in group.cash_effect_summary


def test_authorization_then_capture() -> None:
    parent = make_event(
        event_id="p",
        status=EventStatus.CANCELLED,
        description="Card authorization",
    )
    child = make_event(
        event_id="c",
        status=EventStatus.SETTLED,
        linked_event_id="p",
        description="Settled card purchase",
    )
    assert (
        classify_lifecycle(parent, child).lifecycle_type
        is LifecycleType.AUTHORIZATION_THEN_CAPTURE
    )


def test_purchase_then_pending_refund() -> None:
    parent = make_event(
        event_id="p",
        status=EventStatus.SETTLED,
        description="Purchase awaiting refund",
    )
    child = make_event(
        event_id="c",
        event_type=EventType.REFUND,
        status=EventStatus.PENDING,
        direction=EventDirection.CREDIT,
        linked_event_id="p",
        description="Pending merchant refund",
    )
    assert (
        classify_lifecycle(parent, child).lifecycle_type
        is LifecycleType.PURCHASE_THEN_PENDING_REFUND
    )


def test_failed_then_retry() -> None:
    parent = make_event(
        event_id="p",
        event_type=EventType.DEBT_PAYMENT,
        status=EventStatus.FAILED,
        description="Failed bill payment attempt",
    )
    child = make_event(
        event_id="c",
        event_type=EventType.DEBT_PAYMENT,
        status=EventStatus.SCHEDULED,
        linked_event_id="p",
        description="Scheduled bill payment retry",
    )
    assert classify_lifecycle(parent, child).lifecycle_type is LifecycleType.FAILED_THEN_RETRY


def test_investment_purchase_then_valuation() -> None:
    parent = make_event(
        event_id="p",
        event_type=EventType.INVESTMENT_PURCHASE,
        category="investment",
        description="Investment contribution",
    )
    child = make_event(
        event_id="c",
        event_type=EventType.INVESTMENT_VALUATION,
        status=EventStatus.UNREALIZED,
        direction=EventDirection.NON_CASH,
        linked_event_id="p",
        category="investment",
        description="Current portfolio valuation",
        settlement_date=None,
    )
    assert (
        classify_lifecycle(parent, child).lifecycle_type
        is LifecycleType.INVESTMENT_PURCHASE_THEN_VALUATION
    )


def test_investment_purchase_then_sale() -> None:
    parent = make_event(
        event_id="p",
        event_type=EventType.INVESTMENT_PURCHASE,
        category="investment",
        description="Investment contribution",
    )
    child = make_event(
        event_id="c",
        event_type=EventType.INVESTMENT_SALE,
        direction=EventDirection.CREDIT,
        linked_event_id="p",
        category="investment",
        description="Investment sale proceeds",
    )
    assert (
        classify_lifecycle(parent, child).lifecycle_type
        is LifecycleType.INVESTMENT_PURCHASE_THEN_SALE
    )


def test_possible_duplicate_pending_charge_is_not_auto_discarded() -> None:
    parent = make_event(
        event_id="p",
        status=EventStatus.SETTLED,
        description="Original card charge",
    )
    child = make_event(
        event_id="c",
        status=EventStatus.PENDING,
        linked_event_id="p",
        description="Possible duplicate card charge",
    )
    group = classify_lifecycle(parent, child)
    assert group.lifecycle_type is LifecycleType.POSSIBLE_DUPLICATE_PENDING_CHARGE
    assert "not auto-discarded" in group.cash_effect_summary


def test_unknown_link_is_explicit() -> None:
    parent = make_event(event_id="p", event_type=EventType.INCOME, direction=EventDirection.CREDIT)
    child = make_event(
        event_id="c",
        event_type=EventType.EXPENSE,
        linked_event_id="p",
    )
    assert classify_lifecycle(parent, child).lifecycle_type is LifecycleType.UNKNOWN
