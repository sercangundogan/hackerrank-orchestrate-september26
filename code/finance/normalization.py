"""Classify raw events into a normalized financial state.

This module does not forecast days, attach request payments, or decide
affordability.
"""

from __future__ import annotations

from datetime import date

from data.models import (
    EventDirection,
    EventStatus,
    EventType,
    FinanceRequest,
    FinancialEvent,
    FinancialProfile,
    Flexibility,
)
from data.repository import DatasetRepository
from finance.currency import MissingExchangeRateError, convert_to_home
from finance.lifecycle import build_lifecycles
from finance.models import (
    EventLifecycle,
    IgnoreReason,
    LifecycleRole,
    LifecycleType,
    NormalizedCashEvent,
    NormalizedFinancialState,
)
from finance.recurrence import build_recurring_series_candidates

EXPECTED_DIRECTION = {
    EventType.EXPENSE: EventDirection.DEBIT,
    EventType.SUBSCRIPTION: EventDirection.DEBIT,
    EventType.DEBT_PAYMENT: EventDirection.DEBIT,
    EventType.INCOME: EventDirection.CREDIT,
    EventType.REFUND: EventDirection.CREDIT,
    EventType.INVESTMENT_PURCHASE: EventDirection.DEBIT,
    EventType.INVESTMENT_SALE: EventDirection.CREDIT,
    EventType.INVESTMENT_VALUATION: EventDirection.NON_CASH,
}

_CONFIRMED_SALARY_TOKENS = ("confirmed salary", "confirmed credit")


def finance_request_by_id(repository: DatasetRepository, request_id: str) -> FinanceRequest:
    """Resolve an evaluation or sample *input* request. Never returns sample labels."""
    evaluation = repository.dataset.evaluation_requests_by_id.get(request_id)
    if evaluation is not None:
        return evaluation
    sample = repository.dataset.sample_requests_by_id.get(request_id)
    if sample is not None:
        return sample.request
    raise LookupError(f"unknown request_id {request_id!r}")


def is_confirmed_scheduled_income(event: FinancialEvent) -> bool:
    if event.status is not EventStatus.SCHEDULED:
        return False
    if event.event_type is not EventType.INCOME:
        return False
    if event.direction is not EventDirection.CREDIT:
        return False
    if event.category != "salary":
        return False
    description = event.description.lower()
    return any(token in description for token in _CONFIRMED_SALARY_TOKENS)


def _lifecycle_index(
    events: tuple[FinancialEvent, ...],
    groups: tuple[EventLifecycle, ...],
) -> tuple[dict[str, LifecycleRole], dict[str, LifecycleType]]:
    roles = {event.event_id: LifecycleRole.STANDALONE for event in events}
    types: dict[str, LifecycleType] = {}
    for group in groups:
        roles[group.parent_event_id] = LifecycleRole.PARENT
        roles[group.child_event_id] = LifecycleRole.CHILD
        types[group.parent_event_id] = group.lifecycle_type
        types[group.child_event_id] = group.lifecycle_type
    return roles, types


def classify_raw_event(
    event: FinancialEvent,
    *,
    request_date: date,
    profile: FinancialProfile,
    repository: DatasetRepository,
    lifecycle_role: LifecycleRole,
    lifecycle_type: LifecycleType | None,
) -> NormalizedCashEvent:
    cash_date = event.settlement_date
    requires_evidence = event.amount is None
    ignore_reason: IgnoreReason | None = None
    is_historical = False
    is_prospective = False
    counts_as_cash = False
    amount_home = None

    if event.amount is None:
        ignore_reason = IgnoreReason.BLANK_AMOUNT
    elif event.status is EventStatus.UNREALIZED:
        ignore_reason = IgnoreReason.UNREALIZED
    elif event.status is EventStatus.CANCELLED:
        ignore_reason = IgnoreReason.CANCELLED
    elif event.status is EventStatus.FAILED:
        ignore_reason = IgnoreReason.FAILED
    elif event.settlement_date is None:
        ignore_reason = IgnoreReason.MISSING_SETTLEMENT_DATE
    elif event.status is EventStatus.PENDING and event.direction is EventDirection.CREDIT:
        ignore_reason = IgnoreReason.PENDING_CREDIT
    elif event.status is EventStatus.SCHEDULED and event.direction is EventDirection.CREDIT:
        if is_confirmed_scheduled_income(event):
            is_prospective = True
        else:
            ignore_reason = IgnoreReason.UNCONFIRMED_SCHEDULED_CREDIT
    elif event.status is EventStatus.PENDING and event.direction is EventDirection.DEBIT:
        is_prospective = True
    elif event.status is EventStatus.SCHEDULED and event.direction is EventDirection.DEBIT:
        is_prospective = True
    elif event.status is EventStatus.SETTLED:
        if event.settlement_date < request_date:
            is_historical = True
        else:
            ignore_reason = IgnoreReason.UNEXPECTED_SETTLED_ON_OR_AFTER_REQUEST

    if (
        ignore_reason is None
        and event.amount is not None
        and event.settlement_date is not None
    ):
        try:
            amount_home = convert_to_home(
                event.amount,
                event.currency,
                profile.home_currency,
                event.settlement_date,
                repository,
            )
        except MissingExchangeRateError:
            ignore_reason = IgnoreReason.MISSING_EXCHANGE_RATE
            is_historical = False
            is_prospective = False

    if ignore_reason is None and amount_home is not None and (is_historical or is_prospective):
        counts_as_cash = True

    if event.amount is None:
        # Historical blank rows remain evidence for grouping, but never cash.
        if event.status is EventStatus.SETTLED and event.settlement_date is not None:
            is_historical = event.settlement_date < request_date
        is_prospective = False
        counts_as_cash = False

    return NormalizedCashEvent(
        source_event_id=event.event_id,
        user_id=event.user_id,
        cash_date=cash_date,
        event_date=event.event_date,
        settlement_date=event.settlement_date,
        direction=event.direction,
        amount_home_currency=amount_home,
        original_amount=event.amount,
        original_currency=event.currency,
        event_type=event.event_type,
        category=event.category,
        description=event.description,
        status=event.status,
        lifecycle_role=lifecycle_role,
        lifecycle_type=lifecycle_type,
        flexibility=event.flexibility,
        minimum_allowed_amount=event.minimum_allowed_amount,
        is_historical=is_historical,
        is_prospective=is_prospective,
        counts_as_cash=counts_as_cash,
        ignore_reason=ignore_reason,
        requires_external_evidence=requires_evidence,
    )


def _direction_inconsistencies(events: tuple[FinancialEvent, ...]) -> tuple[str, ...]:
    issues: list[str] = []
    for event in events:
        expected = EXPECTED_DIRECTION.get(event.event_type)
        if expected is not None and event.direction is not expected:
            issues.append(
                f"{event.event_id}: {event.event_type.value} expected "
                f"{expected.value}, found {event.direction.value}"
            )
    return tuple(issues)


def _flexibility_inconsistencies(
    events: tuple[FinancialEvent, ...],
    profile: FinancialProfile,
) -> tuple[str, ...]:
    issues: list[str] = []
    protect = set(profile.expense_categories_to_protect)
    reduce = set(profile.expense_categories_user_is_willing_to_reduce)
    stop = set(profile.expense_categories_user_is_willing_to_stop)
    adjustable = {
        Flexibility.REDUCIBLE,
        Flexibility.STOPPABLE,
        Flexibility.REDUCIBLE_OR_STOPPABLE,
    }
    for event in events:
        if event.flexibility in adjustable and event.category in protect:
            issues.append(
                f"{event.event_id}: {event.flexibility.value} in protected "
                f"category {event.category}"
            )
        if event.flexibility is Flexibility.REDUCIBLE and event.category not in reduce:
            issues.append(
                f"{event.event_id}: reducible category {event.category} not in profile reduce list"
            )
        if event.flexibility is Flexibility.STOPPABLE and event.category not in stop:
            issues.append(
                f"{event.event_id}: stoppable category {event.category} not in profile stop list"
            )
        if event.flexibility is Flexibility.REDUCIBLE_OR_STOPPABLE and (
            event.category not in reduce and event.category not in stop
        ):
            issues.append(
                f"{event.event_id}: reducible_or_stoppable category {event.category} "
                "not in profile reduce/stop lists"
            )
    return tuple(issues)


def normalize_financial_state(
    repository: DatasetRepository,
    request: FinanceRequest,
) -> NormalizedFinancialState:
    profile = repository.profile_for_user(request.user_id)
    events = repository.events_for_user(request.user_id)
    events_by_id = {event.event_id: event for event in events}
    lifecycle_groups = build_lifecycles(events, events_by_id)
    roles, types = _lifecycle_index(events, lifecycle_groups)

    classified = tuple(
        classify_raw_event(
            event,
            request_date=request.request_date,
            profile=profile,
            repository=repository,
            lifecycle_role=roles[event.event_id],
            lifecycle_type=types.get(event.event_id),
        )
        for event in events
    )

    historical = tuple(
        event
        for event in classified
        if event.is_historical and event.counts_as_cash
    )
    pending_debits = tuple(
        event
        for event in classified
        if event.status is EventStatus.PENDING
        and event.direction is EventDirection.DEBIT
        and event.is_prospective
        and event.counts_as_cash
    )
    ignored_pending_credits = tuple(
        event
        for event in classified
        if event.ignore_reason is IgnoreReason.PENDING_CREDIT
    )
    scheduled_debits = tuple(
        event
        for event in classified
        if event.status is EventStatus.SCHEDULED
        and event.direction is EventDirection.DEBIT
        and event.is_prospective
        and event.counts_as_cash
    )
    confirmed_income = tuple(
        event
        for event in classified
        if event.is_prospective
        and event.counts_as_cash
        and event.direction is EventDirection.CREDIT
    )
    ignored = tuple(event for event in classified if event.ignore_reason is not None)
    unresolved = tuple(
        event
        for event in classified
        if event.ignore_reason
        in {
            IgnoreReason.BLANK_AMOUNT,
            IgnoreReason.MISSING_SETTLEMENT_DATE,
            IgnoreReason.MISSING_EXCHANGE_RATE,
            IgnoreReason.UNEXPECTED_SETTLED_ON_OR_AFTER_REQUEST,
        }
    )
    unexpected = tuple(
        event
        for event in classified
        if event.ignore_reason is IgnoreReason.UNEXPECTED_SETTLED_ON_OR_AFTER_REQUEST
    )

    messages = repository.messages_for_user(request.user_id)
    images = repository.images_for_user(request.user_id)
    requires_messages = bool(messages)

    recurring = build_recurring_series_candidates(
        profile=profile,
        historical_events=tuple(event for event in classified if event.is_historical),
        confirmed_scheduled_income=confirmed_income,
        requires_message_resolution=requires_messages,
    )

    return NormalizedFinancialState(
        request_id=request.request_id,
        user_id=request.user_id,
        request_date=request.request_date,
        profile=profile,
        opening_balance=profile.current_available_balance,
        classified_events=classified,
        historical_settled_cash_events=historical,
        pending_debits=pending_debits,
        ignored_pending_credits=ignored_pending_credits,
        scheduled_debits=scheduled_debits,
        confirmed_scheduled_income=confirmed_income,
        ignored_events=ignored,
        unresolved_events=unresolved,
        unexpected_events=unexpected,
        lifecycle_groups=lifecycle_groups,
        recurring_series_candidates=recurring,
        direction_inconsistencies=_direction_inconsistencies(events),
        flexibility_inconsistencies=_flexibility_inconsistencies(events, profile),
        requires_message_resolution=requires_messages,
        user_message_ids=tuple(message.message_id for message in messages),
        user_image_ids=tuple(image.image_id for image in images),
    )
