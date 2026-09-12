from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from data.models import (
    ConsideredPaymentMethod,
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinancialProfile,
    Flexibility,
    Message,
    MessageSourceType,
)
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
)
from evidence.resolver import resolve_conflicts, resolve_financial_state
from finance.adjustments import ForecastAdjustmentKind
from finance.models import (
    IgnoreReason,
    LifecycleRole,
    NormalizedCashEvent,
    NormalizedFinancialState,
)
from tests.factories import FakeRateLookup


def _fact(**overrides: object) -> EvidenceFact:
    values: dict[str, object] = {
        "source_type": EvidenceSourceType.MESSAGE,
        "source_id": "message_a",
        "user_id": "user_test",
        "request_id": "request_test",
        "related_event_id": None,
        "fact_type": FactType.SALARY_AMOUNT_CHANGE,
        "amount": Decimal("100"),
        "currency": Currency.ZAR,
        "effective_date": date(2026, 3, 1),
        "status": FactStatus.ACTIVE,
        "supersedes_event_id": None,
        "confidence": EvidenceConfidence.HIGH,
        "extraction_method": ExtractionMethod.DETERMINISTIC,
        "raw_reference": "message_a",
        "notes": "test",
        "percent": None,
        "category": "salary",
    }
    values.update(overrides)
    return EvidenceFact(**values)  # type: ignore[arg-type]


def _event(**overrides: object) -> NormalizedCashEvent:
    values: dict[str, object] = {
        "source_event_id": "event_blank",
        "user_id": "user_test",
        "cash_date": date(2026, 3, 10),
        "event_date": date(2026, 3, 10),
        "settlement_date": date(2026, 3, 10),
        "direction": EventDirection.DEBIT,
        "amount_home_currency": None,
        "original_amount": None,
        "original_currency": Currency.ZAR,
        "event_type": EventType.EXPENSE,
        "category": "utilities",
        "description": "Outstanding bill",
        "status": EventStatus.SCHEDULED,
        "lifecycle_role": LifecycleRole.STANDALONE,
        "lifecycle_type": None,
        "flexibility": Flexibility.FIXED,
        "minimum_allowed_amount": None,
        "is_historical": False,
        "is_prospective": False,
        "counts_as_cash": False,
        "ignore_reason": IgnoreReason.BLANK_AMOUNT,
        "requires_external_evidence": True,
    }
    values.update(overrides)
    return NormalizedCashEvent(**values)  # type: ignore[arg-type]


def _state(event: NormalizedCashEvent) -> NormalizedFinancialState:
    profile = FinancialProfile(
        user_id="user_test",
        home_currency=Currency.ZAR,
        current_available_balance=Decimal("5000"),
        minimum_balance_to_keep=Decimal("1000"),
        financial_priorities=("rent",),
        expense_categories_to_protect=("rent",),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=(ConsideredPaymentMethod.FULL_PAYMENT,),
        max_installment_months=None,
    )
    return NormalizedFinancialState(
        request_id="request_test",
        user_id="user_test",
        request_date=date(2026, 3, 3),
        profile=profile,
        opening_balance=Decimal("5000"),
        classified_events=(event,),
        historical_settled_cash_events=(),
        pending_debits=(),
        ignored_pending_credits=(),
        scheduled_debits=(),
        confirmed_scheduled_income=(),
        ignored_events=(event,),
        unresolved_events=(event,),
        unexpected_events=(),
        lifecycle_groups=(),
        recurring_series_candidates=(),
        direction_inconsistencies=(),
        flexibility_inconsistencies=(),
        requires_message_resolution=True,
        user_message_ids=("message_a",),
        user_image_ids=(),
    )


def test_newer_same_source_fact_wins() -> None:
    older = _fact(source_id="message_old", amount=Decimal("10"))
    newer = _fact(source_id="message_new", amount=Decimal("20"))
    messages = (
        Message(
            "message_old",
            "user_test",
            None,
            None,
            datetime(2026, 1, 1),
            MessageSourceType.EMPLOYER,
            "old",
        ),
        Message(
            "message_new",
            "user_test",
            None,
            None,
            datetime(2026, 2, 1),
            MessageSourceType.EMPLOYER,
            "new",
        ),
    )
    resolved = resolve_conflicts((older, newer), messages=messages)
    salary = [fact for fact in resolved if fact.fact_type is FactType.SALARY_AMOUNT_CHANGE]
    assert len(salary) == 1
    assert salary[0].amount == Decimal("20")


def test_cancellation_and_safer_facts_are_kept() -> None:
    ended = _fact(fact_type=FactType.EMPLOYMENT_ENDED, amount=None, currency=None)
    unapproved = _fact(fact_type=FactType.UNAPPROVED_INCOME, amount=None, source_id="message_b")
    resolved = resolve_conflicts((ended, unapproved))
    types = {fact.fact_type for fact in resolved}
    assert FactType.EMPLOYMENT_ENDED in types
    assert FactType.UNAPPROVED_INCOME in types


def test_amendment_fills_blank_scheduled_amount() -> None:
    event = _event()
    state = _state(event)
    fact = _fact(
        fact_type=FactType.EVENT_AMOUNT,
        source_type=EvidenceSourceType.IMAGE,
        source_id="image_x",
        related_event_id="event_blank",
        amount=Decimal("80"),
        supersedes_event_id="event_blank",
    )
    resolved = resolve_financial_state(
        state,
        (fact,),
        repository=FakeRateLookup(),
    )
    filled = resolved.classified_events[0]
    assert filled.amount_home_currency == Decimal("80")
    assert filled.requires_external_evidence is False
    assert filled.counts_as_cash is True
    assert filled.is_prospective is True
    assert resolved.unresolved_events == ()
    assert any(
        item.kind is ForecastAdjustmentKind.SALARY_AMOUNT
        for item in resolved.forecast_adjustments
    ) is False
