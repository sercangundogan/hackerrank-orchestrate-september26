from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal

import pytest

from data.models import (
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    Dataset,
)
from data.repository import DatasetRepository
from finance.lifecycle import build_lifecycles
from finance.models import (
    IgnoreReason,
    LifecycleRole,
    LifecycleType,
    UnresolvedCashAmountError,
)
from finance.normalization import (
    EXPECTED_DIRECTION,
    classify_raw_event,
    finance_request_by_id,
    is_confirmed_scheduled_income,
    normalize_financial_state,
)
from tests.factories import FakeRateLookup, make_event


def _classify(event, request_date=date(2024, 3, 3), home=Currency.ZAR, rates=None):
    profile = type(
        "P",
        (),
        {
            "home_currency": home,
            "user_id": "user_01",
            "expense_categories_to_protect": (),
            "expense_categories_user_is_willing_to_reduce": (),
            "expense_categories_user_is_willing_to_stop": (),
        },
    )()
    return classify_raw_event(
        event,
        request_date=request_date,
        profile=profile,
        repository=rates or FakeRateLookup(),
        lifecycle_role=LifecycleRole.STANDALONE,
        lifecycle_type=None,
    )


def test_historical_settled_is_evidence_not_prospective() -> None:
    event = make_event(settlement_date=date(2024, 3, 2), event_date=date(2024, 3, 2))
    classified = _classify(event, request_date=date(2024, 3, 3))
    assert classified.is_historical is True
    assert classified.is_prospective is False
    assert classified.counts_as_cash is True
    assert classified.cash_date == date(2024, 3, 2)
    assert classified.consumable_home_amount() == Decimal("10")


def test_pending_debit_is_prospective_and_pending_credit_is_ignored() -> None:
    debit = make_event(
        event_id="pend_debit",
        status=EventStatus.PENDING,
        settlement_date=date(2024, 3, 5),
        event_date=date(2024, 3, 2),
        description="Pending fuel authorization",
    )
    credit = make_event(
        event_id="pend_credit",
        event_type=EventType.REFUND,
        direction=EventDirection.CREDIT,
        status=EventStatus.PENDING,
        settlement_date=date(2024, 3, 14),
        description="Pending merchant refund",
    )
    classified_debit = _classify(debit)
    classified_credit = _classify(credit)
    assert classified_debit.is_prospective is True
    assert classified_debit.counts_as_cash is True
    assert classified_credit.ignore_reason is IgnoreReason.PENDING_CREDIT
    assert classified_credit.counts_as_cash is False
    assert classified_credit.is_prospective is False


def test_scheduled_debit_and_confirmed_salary() -> None:
    debit = make_event(
        status=EventStatus.SCHEDULED,
        settlement_date=date(2024, 3, 11),
        description="Scheduled school fee",
        category="education",
    )
    salary = make_event(
        event_id="sal",
        event_type=EventType.INCOME,
        direction=EventDirection.CREDIT,
        status=EventStatus.SCHEDULED,
        category="salary",
        description="Next confirmed salary",
        settlement_date=date(2024, 3, 15),
        amount=Decimal("23320"),
    )
    bonus = make_event(
        event_id="bonus",
        event_type=EventType.INCOME,
        direction=EventDirection.CREDIT,
        status=EventStatus.SCHEDULED,
        category="salary",
        description="Unapproved bonus estimate",
        settlement_date=date(2024, 3, 15),
    )
    assert _classify(debit).is_prospective is True
    assert is_confirmed_scheduled_income(salary) is True
    classified_salary = _classify(salary)
    assert classified_salary.is_prospective is True
    assert classified_salary.counts_as_cash is True
    classified_bonus = _classify(bonus)
    assert classified_bonus.ignore_reason is IgnoreReason.UNCONFIRMED_SCHEDULED_CREDIT
    assert classified_bonus.counts_as_cash is False


def test_cancelled_failed_unrealized_are_not_cash() -> None:
    cancelled = _classify(make_event(status=EventStatus.CANCELLED, description="Card authorization"))
    failed = _classify(
        make_event(
            event_type=EventType.DEBT_PAYMENT,
            status=EventStatus.FAILED,
            description="Failed bill payment attempt",
        )
    )
    unrealized = _classify(
        make_event(
            event_type=EventType.INVESTMENT_VALUATION,
            status=EventStatus.UNREALIZED,
            direction=EventDirection.NON_CASH,
            settlement_date=None,
            description="Current portfolio valuation",
        )
    )
    assert cancelled.ignore_reason is IgnoreReason.CANCELLED
    assert failed.ignore_reason is IgnoreReason.FAILED
    assert unrealized.ignore_reason is IgnoreReason.UNREALIZED
    for item in (cancelled, failed, unrealized):
        assert item.counts_as_cash is False
        assert item.is_prospective is False
        with pytest.raises(UnresolvedCashAmountError):
            item.consumable_home_amount()


def test_blank_amount_is_unresolved_and_never_zero() -> None:
    event = make_event(amount=None, description="August 2019 net salary")
    classified = _classify(event, request_date=date(2019, 9, 3))
    assert classified.original_amount is None
    assert classified.amount_home_currency is None
    assert classified.requires_external_evidence is True
    assert classified.counts_as_cash is False
    assert classified.ignore_reason is IgnoreReason.BLANK_AMOUNT
    with pytest.raises(UnresolvedCashAmountError):
        classified.consumable_home_amount()


def test_unexpected_settled_on_or_after_request_is_unresolved() -> None:
    event = make_event(settlement_date=date(2024, 3, 3), event_date=date(2024, 3, 3))
    classified = _classify(event, request_date=date(2024, 3, 3))
    assert classified.ignore_reason is IgnoreReason.UNEXPECTED_SETTLED_ON_OR_AFTER_REQUEST
    assert classified.counts_as_cash is False
    assert classified.is_historical is False


def test_fx_conversion_and_missing_rate_during_classification() -> None:
    event = make_event(
        amount=Decimal("1800"),
        currency=Currency.USD,
        settlement_date=date(2023, 10, 15),
        event_date=date(2023, 10, 15),
        event_type=EventType.INCOME,
        direction=EventDirection.CREDIT,
        category="salary",
        description="International employer payroll",
    )
    rates = FakeRateLookup(
        {(date(2023, 10, 15), Currency.USD, Currency.IDR): Decimal("15833.33")}
    )
    classified = _classify(
        event, request_date=date(2024, 3, 6), home=Currency.IDR, rates=rates
    )
    assert classified.amount_home_currency == Decimal("1800") * Decimal("15833.33")
    missing = _classify(event, request_date=date(2024, 3, 6), home=Currency.IDR)
    assert missing.ignore_reason is IgnoreReason.MISSING_EXCHANGE_RATE
    assert missing.counts_as_cash is False


def test_real_dataset_lifecycle_and_cash_invariants(
    dataset: Dataset, repository: DatasetRepository
) -> None:
    linked = [event for event in dataset.events if event.linked_event_id]
    groups = build_lifecycles(tuple(linked), dataset.events_by_id)
    assert len(groups) == 58
    assert LifecycleType.UNKNOWN not in {group.lifecycle_type for group in groups}
    counts = Counter(group.lifecycle_type for group in groups)
    assert counts[LifecycleType.CHARGE_THEN_REFUND] == 14
    assert counts[LifecycleType.INVESTMENT_PURCHASE_THEN_VALUATION] == 10
    assert counts[LifecycleType.AUTHORIZATION_THEN_CAPTURE] == 8
    assert counts[LifecycleType.PURCHASE_THEN_PENDING_REFUND] == 8
    assert counts[LifecycleType.FAILED_THEN_RETRY] == 7
    assert counts[LifecycleType.POSSIBLE_DUPLICATE_PENDING_CHARGE] == 6
    assert counts[LifecycleType.INVESTMENT_PURCHASE_THEN_SALE] == 5

    for event in dataset.events:
        assert event.direction is EXPECTED_DIRECTION[event.event_type]

    all_requests = [sample.request for sample in dataset.sample_requests]
    all_requests.extend(dataset.evaluation_requests)
    blank_unresolved = 0
    missing_fx = 0
    for request in all_requests:
        state = normalize_financial_state(repository, request)
        assert state.direction_inconsistencies == ()
        assert state.flexibility_inconsistencies == ()
        assert state.unexpected_events == ()
        assert state.opening_balance == state.profile.current_available_balance
        for event in state.classified_events:
            if event.status is EventStatus.UNREALIZED:
                assert event.counts_as_cash is False
            if event.ignore_reason in {IgnoreReason.CANCELLED, IgnoreReason.FAILED}:
                assert event.is_prospective is False
                assert event.counts_as_cash is False
            if event.requires_external_evidence:
                assert event.original_amount is None
                assert event.amount_home_currency is None
                assert event.counts_as_cash is False
                blank_unresolved += 1
            if (
                event.original_amount is not None
                and event.original_currency is not state.profile.home_currency
                and event.settlement_date is not None
                and event.ignore_reason is not IgnoreReason.UNREALIZED
            ):
                if event.ignore_reason is IgnoreReason.MISSING_EXCHANGE_RATE:
                    missing_fx += 1
                elif event.counts_as_cash or event.is_historical:
                    assert event.amount_home_currency is not None
        for event in state.prospective_cash_events():
            event.consumable_home_amount()
    assert blank_unresolved == 16
    assert missing_fx == 0


def test_sample_normalization_does_not_use_labels(
    dataset: Dataset, repository: DatasetRepository
) -> None:
    sample = dataset.sample_requests_by_id["request_01"]
    request = finance_request_by_id(repository, "request_01")
    assert request is sample.request
    state = normalize_financial_state(repository, request)
    assert not hasattr(state, "amount_safe_to_pay")
    pending = {event.source_event_id for event in state.pending_debits}
    assert "event_102" in pending
    income_ids = {event.source_event_id for event in state.confirmed_scheduled_income}
    assert "event_103" in income_ids
    cancelled = [
        event
        for event in state.classified_events
        if event.source_event_id == "event_100"
    ][0]
    assert cancelled.counts_as_cash is False
    captured = [
        event
        for event in state.classified_events
        if event.source_event_id == "event_101"
    ][0]
    assert captured.is_historical is True
    refund = [
        event
        for event in state.classified_events
        if event.source_event_id == "event_99"
    ][0]
    charge = [
        event
        for event in state.classified_events
        if event.source_event_id == "event_98"
    ][0]
    assert charge.counts_as_cash is True
    assert refund.counts_as_cash is True


def test_request_11_sparse_dining_remains_a_candidate(
    repository: DatasetRepository,
) -> None:
    state = normalize_financial_state(
        repository, finance_request_by_id(repository, "request_11")
    )
    dining = [
        series
        for series in state.recurring_series_candidates
        if "weekend food delivery" in series.normalized_description
    ]
    assert dining
    assert dining[0].likely_recurring is True
    assert dining[0].flexibility.value == "reducible"
    assert dining[0].minimum_allowed_amount == Decimal("665950")


def test_request_03_blank_salary_stays_unresolved(repository: DatasetRepository) -> None:
    state = normalize_financial_state(
        repository, finance_request_by_id(repository, "request_03")
    )
    blank = [event for event in state.unresolved_events if event.source_event_id == "event_253"]
    assert blank
    assert blank[0].requires_external_evidence is True
    assert blank[0].counts_as_cash is False
