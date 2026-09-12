"""Apply EvidenceFacts to a copy of the normalized state.

Raw Phase 1 objects are never mutated. Conflict order follows the problem
statement: explicit amendment, then newer same-source record, then settled
over estimate, then the financially safer reading.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from data.models import EventDirection, EventStatus, Message
from data.repository import DatasetRepository
from evidence.models import EvidenceFact, EvidenceSourceType, FactType
from finance.adjustments import ForecastAdjustment, ForecastAdjustmentKind
from finance.currency import MissingExchangeRateError, convert_to_home
from finance.models import (
    IgnoreReason,
    NormalizedCashEvent,
    NormalizedFinancialState,
)
from finance.recurrence import build_recurring_series_candidates

_AMENDMENTS = {
    FactType.EMPLOYMENT_ENDED,
    FactType.SALARY_AMOUNT_CHANGE,
    FactType.SALARY_PAYMENT_DATE_CHANGE,
    FactType.TEMPORARY_SALARY_CHANGE,
    FactType.RECURRING_EXPENSE_CHANGE,
    FactType.EVENT_AMOUNT,
}

_SAFER = {
    FactType.UNAPPROVED_INCOME,
    FactType.IGNORE_CREDIT,
    FactType.SCAM_OR_UNTRUSTED_PAYMENT_REQUEST,
    FactType.REFUND_STILL_PENDING,
}


def resolve_conflicts(
    facts: tuple[EvidenceFact, ...],
    *,
    messages: tuple[Message, ...] = (),
) -> tuple[EvidenceFact, ...]:
    sent_at = {message.message_id: message.sent_at for message in messages}

    def rank(fact: EvidenceFact) -> tuple:
        explicit = 1 if fact.fact_type in _AMENDMENTS else 0
        when = sent_at.get(fact.source_id)
        ts = when.timestamp() if when is not None else 0.0
        safer = 1 if fact.fact_type in _SAFER else 0
        image = 1 if fact.source_type is EvidenceSourceType.IMAGE else 0
        return (explicit, ts, safer, image, fact.source_id, fact.fact_type.value)

    ordered = sorted(facts, key=rank)
    kept: dict[tuple[str, str, str | None], EvidenceFact] = {}
    passthrough: list[EvidenceFact] = []
    for fact in ordered:
        if fact.fact_type in {
            FactType.EVENT_AMOUNT,
            FactType.SCAM_OR_UNTRUSTED_PAYMENT_REQUEST,
            FactType.OTHER_RELEVANT_FINANCIAL_FACT,
            FactType.PAYMENT_RETRY_CONFIRMED,
            FactType.REFUND_STILL_PENDING,
            FactType.IGNORE_CREDIT,
            FactType.UNAPPROVED_INCOME,
            FactType.CONFIRMED_FUTURE_INCOME,
            FactType.RECURRING_EXPENSE_ADDED,
        }:
            passthrough.append(fact)
            continue
        key = (fact.user_id, fact.fact_type.value, fact.category)
        kept[key] = fact
    return tuple(list(kept.values()) + passthrough)


def _fill_amount(
    event: NormalizedCashEvent,
    fact: EvidenceFact,
    state: NormalizedFinancialState,
    repository: DatasetRepository,
) -> NormalizedCashEvent | None:
    if fact.amount is None:
        return None
    cash_date = event.settlement_date or event.event_date
    try:
        home = convert_to_home(
            fact.amount,
            fact.currency or event.original_currency,
            state.profile.home_currency,
            cash_date,
            repository,
        )
    except MissingExchangeRateError:
        return None

    ignore_reason = None
    is_historical = event.is_historical
    is_prospective = event.is_prospective
    if event.status is EventStatus.PENDING and event.direction is EventDirection.DEBIT:
        is_prospective = True
        is_historical = False
    elif event.status is EventStatus.SCHEDULED and event.direction is EventDirection.DEBIT:
        is_prospective = True
        is_historical = False
    elif event.status is EventStatus.SETTLED and event.settlement_date is not None:
        is_historical = event.settlement_date < state.request_date
        is_prospective = False
    elif event.status is EventStatus.PENDING and event.direction is EventDirection.CREDIT:
        ignore_reason = IgnoreReason.PENDING_CREDIT
        is_prospective = False

    counts = ignore_reason is None and (is_historical or is_prospective)
    return replace(
        event,
        amount_home_currency=home,
        original_amount=fact.amount,
        ignore_reason=ignore_reason,
        requires_external_evidence=False,
        is_historical=is_historical,
        is_prospective=is_prospective,
        counts_as_cash=counts,
    )


def _home_amount(
    fact: EvidenceFact,
    state: NormalizedFinancialState,
    repository: DatasetRepository,
) -> Decimal | None:
    if fact.amount is None or fact.currency is None:
        return fact.amount
    if fact.currency is state.profile.home_currency:
        return fact.amount
    when = fact.effective_date or state.request_date
    try:
        return convert_to_home(
            fact.amount,
            fact.currency,
            state.profile.home_currency,
            when,
            repository,
        )
    except MissingExchangeRateError:
        return None


def _adjustments_from_facts(
    facts: tuple[EvidenceFact, ...],
    state: NormalizedFinancialState,
    repository: DatasetRepository,
) -> tuple[ForecastAdjustment, ...]:
    adjustments: list[ForecastAdjustment] = []
    for fact in facts:
        source = (fact.source_id,)
        if fact.fact_type is FactType.EMPLOYMENT_ENDED:
            remaining = any(
                other.source_id == fact.source_id
                and other.fact_type is FactType.SALARY_AMOUNT_CHANGE
                for other in facts
            )
            if remaining:
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.STOP_SALARY_PROJECTION,
                    source_ids=source,
                    notes=fact.notes,
                )
            )
        elif fact.fact_type is FactType.SALARY_AMOUNT_CHANGE:
            home = _home_amount(fact, state, repository)
            if home is None:
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.SALARY_AMOUNT,
                    amount_home=home,
                    effective_date=fact.effective_date or state.request_date,
                    source_ids=source,
                    notes=fact.notes,
                    income_subtype="base_salary",
                    category=fact.category or "salary",
                )
            )
        elif fact.fact_type is FactType.TEMPORARY_SALARY_CHANGE:
            home = _home_amount(fact, state, repository)
            if home is None:
                continue
            adjustments.append(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.SALARY_TEMPORARY,
                    amount_home=home,
                    effective_date=fact.effective_date or state.request_date,
                    occurrences=1,
                    source_ids=source,
                    notes=fact.notes,
                    income_subtype="base_salary",
                    category=fact.category or "salary",
                )
            )
        elif fact.fact_type is FactType.SALARY_PAYMENT_DATE_CHANGE and fact.effective_date:
            adjustments.append(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.SALARY_PAYDAY,
                    effective_date=fact.effective_date,
                    source_ids=source,
                    notes=fact.notes,
                    income_subtype="base_salary",
                    category=fact.category or "salary",
                )
            )
        elif fact.fact_type is FactType.CONFIRMED_FUTURE_INCOME and fact.amount is not None:
            home = _home_amount(fact, state, repository)
            if home is None or fact.effective_date is None:
                continue
            kind = (
                ForecastAdjustmentKind.START_SALARY
                if fact.category == "salary"
                else ForecastAdjustmentKind.CONFIRM_INCOME
            )
            adjustments.append(
                ForecastAdjustment(
                    kind=kind,
                    amount_home=home,
                    effective_date=fact.effective_date,
                    category=fact.category,
                    source_ids=source,
                    notes=fact.notes,
                    income_subtype="base_salary" if fact.category == "salary" else None,
                )
            )
        elif fact.fact_type is FactType.RECURRING_EXPENSE_CHANGE and fact.percent is not None:
            adjustments.append(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.RENT_PERCENT,
                    percent=fact.percent,
                    effective_date=fact.effective_date or state.request_date,
                    category=fact.category or "rent",
                    source_ids=source,
                    notes=fact.notes,
                )
            )
    return tuple(adjustments)


def _partition(classified: tuple[NormalizedCashEvent, ...]) -> dict[str, tuple[NormalizedCashEvent, ...]]:
    historical = tuple(
        event for event in classified if event.is_historical and event.counts_as_cash
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
        if event.is_prospective and event.counts_as_cash and event.direction is EventDirection.CREDIT
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
        or (event.requires_external_evidence and event.amount_home_currency is None)
    )
    unexpected = tuple(
        event
        for event in classified
        if event.ignore_reason is IgnoreReason.UNEXPECTED_SETTLED_ON_OR_AFTER_REQUEST
    )
    return {
        "historical": historical,
        "pending_debits": pending_debits,
        "ignored_pending_credits": ignored_pending_credits,
        "scheduled_debits": scheduled_debits,
        "confirmed_income": confirmed_income,
        "ignored": ignored,
        "unresolved": unresolved,
        "unexpected": unexpected,
    }


def resolve_financial_state(
    normalized_state: NormalizedFinancialState,
    evidence_facts: tuple[EvidenceFact, ...],
    *,
    repository: DatasetRepository,
    messages: tuple[Message, ...] = (),
) -> NormalizedFinancialState:
    facts = resolve_conflicts(evidence_facts, messages=messages)
    classified = list(normalized_state.classified_events)
    by_id = {event.source_event_id: index for index, event in enumerate(classified)}
    for fact in facts:
        if fact.fact_type is not FactType.EVENT_AMOUNT or not fact.related_event_id:
            continue
        index = by_id.get(fact.related_event_id)
        if index is None:
            continue
        filled = _fill_amount(classified[index], fact, normalized_state, repository)
        if filled is not None:
            classified[index] = filled
    classified_tuple = tuple(classified)
    parts = _partition(classified_tuple)
    salary_resolved = any(
        fact.fact_type
        in {
            FactType.SALARY_AMOUNT_CHANGE,
            FactType.TEMPORARY_SALARY_CHANGE,
            FactType.EMPLOYMENT_ENDED,
            FactType.SALARY_PAYMENT_DATE_CHANGE,
            FactType.CONFIRMED_FUTURE_INCOME,
        }
        for fact in facts
    )
    recurring = build_recurring_series_candidates(
        profile=normalized_state.profile,
        historical_events=tuple(event for event in classified_tuple if event.is_historical),
        confirmed_scheduled_income=parts["confirmed_income"],
        requires_message_resolution=bool(normalized_state.user_message_ids)
        and not salary_resolved,
    )
    adjustments = _adjustments_from_facts(facts, normalized_state, repository)
    return replace(
        normalized_state,
        classified_events=classified_tuple,
        historical_settled_cash_events=parts["historical"],
        pending_debits=parts["pending_debits"],
        ignored_pending_credits=parts["ignored_pending_credits"],
        scheduled_debits=parts["scheduled_debits"],
        confirmed_scheduled_income=parts["confirmed_income"],
        ignored_events=parts["ignored"],
        unresolved_events=parts["unresolved"],
        unexpected_events=parts["unexpected"],
        recurring_series_candidates=recurring,
        requires_message_resolution=bool(normalized_state.user_message_ids)
        and not salary_resolved,
        forecast_adjustments=adjustments,
        evidence_source_ids=tuple(dict.fromkeys(fact.source_id for fact in facts)),
    )
