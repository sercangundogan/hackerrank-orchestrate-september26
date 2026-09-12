from __future__ import annotations

from datetime import date
from decimal import Decimal

from data.models import EventDirection, EventStatus, EventType
from finance.adjustments import ForecastAdjustment, ForecastAdjustmentKind
from finance.forecast import forecast_financial_state
from finance.forecast_models import ForecastConfig, ForecastEventKind, SalaryProjectionMode
from finance.income import IncomeSubtype, classify_income_text
from finance.models import Cadence, CadenceConfidence, IgnoreReason
from tests.test_forecast import _event, _series, _state


def _payroll_series(**overrides: object):
    values = dict(
        category="salary",
        normalized_description="payroll credit",
        original_description="Payroll credit",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        inferred_cadence_days=30,
        cadence_confidence=CadenceConfidence.HIGH,
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
    values.update(overrides)
    return _series(**values)


def _salary_credits(result):
    return [
        entry
        for entry in result.all_entries
        if entry.signed_amount > 0 and entry.category == "salary"
    ]


def test_classify_commission_and_bonus() -> None:
    assert classify_income_text("Monthly sales commission", "salary") is IncomeSubtype.COMMISSION
    assert classify_income_text("Performance commission", "salary") is IncomeSubtype.COMMISSION
    assert classify_income_text("Quarterly bonus", "salary") is IncomeSubtype.BONUS
    assert classify_income_text("Payroll credit", "salary") is IncomeSubtype.BASE_SALARY
    assert classify_income_text("Final employer payroll", "salary") is IncomeSubtype.FINAL_PAYROLL


def test_history_only_base_salary_not_generated_under_confirmed_then_continue() -> None:
    result = forecast_financial_state(
        _state(recurring_series_candidates=(_payroll_series(),)),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.CONFIRMED_THEN_CONTINUE),
    )
    assert _salary_credits(result) == []


def test_scheduled_confirmed_salary_counts_once_and_may_continue() -> None:
    series = _payroll_series(scheduled_confirmed_event_ids=("sal_sched",))
    scheduled = _event(
        source_event_id="sal_sched",
        cash_date=date(2026, 3, 15),
        event_date=date(2026, 3, 1),
        settlement_date=date(2026, 3, 15),
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        status=EventStatus.SCHEDULED,
        amount_home_currency=Decimal("3000"),
    )
    state = _state(
        confirmed_scheduled_income=(scheduled,),
        classified_events=(scheduled,),
        recurring_series_candidates=(series,),
    )
    result = forecast_financial_state(
        state,
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.CONFIRMED_THEN_CONTINUE),
    )
    march = [entry for entry in _salary_credits(result) if entry.date == date(2026, 3, 15)]
    assert len(march) == 1
    assert march[0].kind is ForecastEventKind.CONFIRMED_CREDIT
    assert march[0].explicitly_scheduled is True
    later = [entry for entry in _salary_credits(result) if entry.is_generated_recurrence]
    assert later
    assert all(entry.date != date(2026, 3, 15) for entry in later)


def test_strict_confirmed_ignores_history_only_salary() -> None:
    result = forecast_financial_state(
        _state(recurring_series_candidates=(_payroll_series(),)),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.STRICT_CONFIRMED),
    )
    assert _salary_credits(result) == []


def test_confirmed_salary_message_creates_dated_income() -> None:
    result = forecast_financial_state(
        _state(
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.START_SALARY,
                    amount_home=Decimal("1800"),
                    effective_date=date(2026, 3, 20),
                    category="salary",
                    source_ids=("message_job",),
                    notes="first salary",
                    income_subtype="base_salary",
                ),
            )
        ),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.STRICT_CONFIRMED),
    )
    credits = _salary_credits(result)
    assert len(credits) == 1
    assert credits[0].date == date(2026, 3, 20)
    assert credits[0].message_confirmed is True


def test_salary_raise_targets_base_salary_only() -> None:
    base = _payroll_series()
    commission = _series(
        category="salary",
        normalized_description="performance commission",
        original_description="Performance commission",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        cadence_confidence=CadenceConfidence.MEDIUM,
        observed_dates=(date(2025, 12, 24), date(2026, 1, 24), date(2026, 2, 24)),
        event_ids=("c1", "c2", "c3"),
        representative_amount=Decimal("900"),
        observed_amounts_home_currency=(Decimal("900"),) * 3,
    )
    result = forecast_financial_state(
        _state(
            recurring_series_candidates=(base, commission),
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.SALARY_AMOUNT,
                    amount_home=Decimal("4000"),
                    effective_date=date(2026, 3, 15),
                    source_ids=("message_raise",),
                    notes="raise",
                    income_subtype="base_salary",
                ),
            ),
        ),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.CONFIRMED_THEN_CONTINUE),
    )
    credits = _salary_credits(result)
    assert credits
    assert all(entry.income_subtype == "base_salary" for entry in credits)
    assert all(entry.amount_home_currency == Decimal("4000") for entry in credits)
    assert not any("commission" in entry.description.lower() for entry in credits)


def test_commission_history_is_not_projected() -> None:
    commission = _series(
        category="salary",
        normalized_description="monthly sales commission",
        original_description="Monthly sales commission",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        cadence_confidence=CadenceConfidence.HIGH,
        observed_dates=(date(2025, 12, 24), date(2026, 1, 24), date(2026, 2, 24)),
        event_ids=("c1", "c2", "c3"),
        representative_amount=Decimal("1200"),
        observed_amounts_home_currency=(Decimal("1200"),) * 3,
    )
    result = forecast_financial_state(
        _state(recurring_series_candidates=(commission,)),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.STRONG_BASE_SALARY),
    )
    assert _salary_credits(result) == []


def test_bonus_history_is_not_projected() -> None:
    bonus = _series(
        category="salary",
        normalized_description="quarterly bonus",
        original_description="Quarterly bonus",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        cadence_confidence=CadenceConfidence.HIGH,
        observed_dates=(date(2025, 11, 1), date(2025, 12, 1), date(2026, 1, 1)),
        event_ids=("b1", "b2", "b3"),
        representative_amount=Decimal("500"),
        observed_amounts_home_currency=(Decimal("500"),) * 3,
    )
    result = forecast_financial_state(_state(recurring_series_candidates=(bonus,)))
    assert _salary_credits(result) == []


def test_pending_refund_is_ignored() -> None:
    refund = _event(
        source_event_id="refund_1",
        direction=EventDirection.CREDIT,
        category="shopping",
        description="Store refund",
        status=EventStatus.PENDING,
        cash_date=date(2026, 3, 10),
        settlement_date=date(2026, 3, 10),
        is_historical=False,
        is_prospective=False,
        counts_as_cash=False,
        ignore_reason=IgnoreReason.PENDING_CREDIT,
        amount_home_currency=Decimal("200"),
    )
    result = forecast_financial_state(
        _state(
            classified_events=(refund,),
            ignored_pending_credits=(refund,),
            ignored_events=(refund,),
        )
    )
    assert not any(entry.signed_amount > 0 for entry in result.all_entries)
    assert any("pending credit" in line for line in result.ignored_summaries)


def test_employment_ended_stops_salary() -> None:
    result = forecast_financial_state(
        _state(
            recurring_series_candidates=(_payroll_series(),),
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.STOP_SALARY_PROJECTION,
                    source_ids=("message_end",),
                    notes="employment ended",
                ),
            ),
        )
    )
    assert _salary_credits(result) == []


def test_final_payroll_is_not_continued() -> None:
    series = _payroll_series(
        normalized_description="final employer payroll",
        original_description="Final employer payroll",
        cadence_confidence=CadenceConfidence.HIGH,
    )
    result = forecast_financial_state(
        _state(recurring_series_candidates=(series,)),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.STRONG_BASE_SALARY),
    )
    assert _salary_credits(result) == []


def test_payday_move_applies_to_base_payroll() -> None:
    result = forecast_financial_state(
        _state(
            recurring_series_candidates=(_payroll_series(),),
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.SALARY_PAYDAY,
                    effective_date=date(2026, 3, 23),
                    source_ids=("message_payday",),
                    notes="payday moved",
                    income_subtype="base_salary",
                ),
            ),
        ),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.CONFIRMED_THEN_CONTINUE),
    )
    dates = {entry.date for entry in _salary_credits(result)}
    assert date(2026, 3, 23) in dates
    assert date(2026, 3, 15) not in dates


def test_strong_base_salary_history_may_continue() -> None:
    result = forecast_financial_state(
        _state(recurring_series_candidates=(_payroll_series(),)),
        ForecastConfig(salary_projection_mode=SalaryProjectionMode.STRONG_BASE_SALARY),
    )
    credits = _salary_credits(result)
    assert credits
    assert all(entry.income_subtype == "base_salary" for entry in credits)
    assert all(entry.generated_from_history for entry in credits)
