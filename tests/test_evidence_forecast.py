from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from data.models import Currency, EventDirection, EventStatus, EventType
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
)
from evidence.resolver import resolve_financial_state
from finance.adjustments import ForecastAdjustment, ForecastAdjustmentKind
from finance.forecast import forecast_financial_state
from finance.forecast_models import ForecastConfig, UnresolvedForecastError
from finance.models import Cadence, IgnoreReason
from tests.factories import FakeRateLookup
from tests.test_forecast import _event, _series, _state


def test_salary_amendment_changes_generated_future_salary() -> None:
    series = _series(
        category="salary",
        normalized_description="payroll credit",
        original_description="Payroll credit",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        inferred_cadence_days=30,
        observed_dates=(
            date(2025, 11, 15),
            date(2025, 12, 15),
            date(2026, 1, 15),
            date(2026, 2, 15),
        ),
        event_ids=("s1", "s2", "s3", "s4"),
        representative_amount=Decimal("3000"),
        observed_amounts_home_currency=(Decimal("3000"),) * 4,
    )
    before = forecast_financial_state(_state(recurring_series_candidates=(series,)))
    after = forecast_financial_state(
        _state(
            recurring_series_candidates=(series,),
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.SALARY_AMOUNT,
                    amount_home=Decimal("4000"),
                    effective_date=date(2026, 3, 15),
                    source_ids=("message_raise",),
                    notes="raise",
                ),
            ),
        )
    )
    before_amt = next(
        entry.amount_home_currency
        for entry in before.all_entries
        if entry.category == "salary" and entry.date == date(2026, 3, 15)
    )
    after_amt = next(
        entry.amount_home_currency
        for entry in after.all_entries
        if entry.category == "salary" and entry.date == date(2026, 3, 15)
    )
    assert before_amt == Decimal("3000")
    assert after_amt == Decimal("4000")


def test_employment_end_stops_generated_salary() -> None:
    series = _series(
        category="salary",
        normalized_description="payroll credit",
        original_description="Payroll credit",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        observed_dates=(
            date(2025, 11, 15),
            date(2025, 12, 15),
            date(2026, 1, 15),
            date(2026, 1, 15),
            date(2026, 2, 15),
        ),
        event_ids=("s1", "s2", "s3", "s4"),
        representative_amount=Decimal("3000"),
        observed_amounts_home_currency=(Decimal("3000"),) * 4,
    )
    result = forecast_financial_state(
        _state(
            recurring_series_candidates=(series,),
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.STOP_SALARY_PROJECTION,
                    source_ids=("message_end",),
                    notes="employment ended",
                ),
            ),
        )
    )
    assert not any(entry.category == "salary" for entry in result.all_entries)


def test_resolved_blank_amount_enters_forecast() -> None:
    blank = _event(
        source_event_id="event_blank",
        amount_home_currency=None,
        original_amount=None,
        status=EventStatus.SCHEDULED,
        cash_date=date(2026, 3, 20),
        settlement_date=date(2026, 3, 20),
        event_date=date(2026, 3, 20),
        requires_external_evidence=True,
        is_prospective=False,
        counts_as_cash=False,
        ignore_reason=IgnoreReason.BLANK_AMOUNT,
        description="Outstanding bill",
        category="utilities",
    )
    state = _state(
        classified_events=(blank,),
        unresolved_events=(blank,),
        ignored_events=(blank,),
        requires_message_resolution=False,
    )
    fact = EvidenceFact(
        source_type=EvidenceSourceType.IMAGE,
        source_id="image_x",
        user_id="user_test",
        request_id="request_test",
        related_event_id="event_blank",
        fact_type=FactType.EVENT_AMOUNT,
        amount=Decimal("80"),
        currency=Currency.ZAR,
        effective_date=None,
        status=FactStatus.ACTIVE,
        supersedes_event_id="event_blank",
        confidence=EvidenceConfidence.HIGH,
        extraction_method=ExtractionMethod.VLM,
        raw_reference="image_x",
        notes="balance due",
    )
    resolved = resolve_financial_state(state, (fact,), repository=FakeRateLookup())
    result = forecast_financial_state(resolved)
    applied = [entry for entry in result.all_entries if entry.source_event_id == "event_blank"]
    assert applied
    assert applied[0].amount_home_currency == Decimal("80")
    assert applied[0].date == date(2026, 3, 20)


def test_unresolved_critical_evidence_fails_strict_mode() -> None:
    blank = _event(
        source_event_id="event_blank",
        amount_home_currency=None,
        original_amount=None,
        requires_external_evidence=True,
        is_prospective=True,
        counts_as_cash=False,
        ignore_reason=IgnoreReason.BLANK_AMOUNT,
    )
    state = _state(unresolved_events=(blank,), classified_events=(blank,))
    with pytest.raises(UnresolvedForecastError):
        forecast_financial_state(state, ForecastConfig(strict_unresolved_amounts=True))
