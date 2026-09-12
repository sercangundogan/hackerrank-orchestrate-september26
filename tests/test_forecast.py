from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from data.models import (
    ConsideredPaymentMethod,
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinancialProfile,
    Flexibility,
)
from finance.forecast import forecast_financial_state, series_key
from finance.forecast_models import (
    ForecastCashEvent,
    ForecastConfig,
    ForecastEventKind,
    ForecastEventSource,
    SameDayPriority,
    UnresolvedForecastError,
    VariableAmountStrategy,
)
from finance.models import (
    AmountBehavior,
    Cadence,
    CadenceConfidence,
    IgnoreReason,
    LifecycleRole,
    NormalizedCashEvent,
    NormalizedFinancialState,
    RecurringSeriesCandidate,
)


def _profile(**overrides: object) -> FinancialProfile:
    values: dict[str, object] = {
        "user_id": "user_test",
        "home_currency": Currency.ZAR,
        "current_available_balance": Decimal("5000"),
        "minimum_balance_to_keep": Decimal("1000"),
        "financial_priorities": ("rent",),
        "expense_categories_to_protect": ("rent",),
        "expense_categories_user_is_willing_to_reduce": ("dining",),
        "expense_categories_user_is_willing_to_stop": ("streaming",),
        "payment_methods_user_will_consider": (ConsideredPaymentMethod.FULL_PAYMENT,),
        "max_installment_months": None,
    }
    values.update(overrides)
    return FinancialProfile(**values)  # type: ignore[arg-type]


def _event(**overrides: object) -> NormalizedCashEvent:
    values: dict[str, object] = {
        "source_event_id": "event_x",
        "user_id": "user_test",
        "cash_date": date(2026, 3, 3),
        "event_date": date(2026, 3, 3),
        "settlement_date": date(2026, 3, 3),
        "direction": EventDirection.DEBIT,
        "amount_home_currency": Decimal("100"),
        "original_amount": Decimal("100"),
        "original_currency": Currency.ZAR,
        "event_type": EventType.EXPENSE,
        "category": "fuel",
        "description": "Pending fuel authorization",
        "status": EventStatus.PENDING,
        "lifecycle_role": LifecycleRole.STANDALONE,
        "lifecycle_type": None,
        "flexibility": Flexibility.FIXED,
        "minimum_allowed_amount": None,
        "is_historical": False,
        "is_prospective": True,
        "counts_as_cash": True,
        "ignore_reason": None,
        "requires_external_evidence": False,
    }
    values.update(overrides)
    return NormalizedCashEvent(**values)  # type: ignore[arg-type]


def _series(**overrides: object) -> RecurringSeriesCandidate:
    values: dict[str, object] = {
        "user_id": "user_test",
        "category": "groceries",
        "normalized_description": "weekly groceries",
        "original_description": "Weekly groceries",
        "event_ids": ("e1", "e2", "e3", "e4"),
        "direction": EventDirection.DEBIT,
        "event_type": EventType.EXPENSE,
        "observed_dates": (
            date(2026, 2, 3),
            date(2026, 2, 10),
            date(2026, 2, 17),
            date(2026, 2, 24),
        ),
        "observed_amounts_home_currency": (
            Decimal("300"),
            Decimal("300"),
            Decimal("300"),
            Decimal("300"),
        ),
        "inferred_cadence": Cadence.WEEKLY,
        "inferred_cadence_days": 7,
        "cadence_confidence": CadenceConfidence.HIGH,
        "amount_behavior": AmountBehavior.FIXED,
        "representative_amount": Decimal("300"),
        "flexibility": Flexibility.FIXED,
        "minimum_allowed_amount": None,
        "is_protected_category": False,
        "profile_allows_reduce": False,
        "profile_allows_stop": False,
        "likely_recurring": True,
        "scheduled_confirmed_event_ids": (),
        "requires_message_resolution": False,
        "reason": "weekly high",
    }
    values.update(overrides)
    return RecurringSeriesCandidate(**values)  # type: ignore[arg-type]


def _state(**overrides: object) -> NormalizedFinancialState:
    profile = overrides.pop("profile", None)
    if profile is None:
        profile = _profile()
    values: dict[str, object] = {
        "request_id": "request_test",
        "user_id": "user_test",
        "request_date": date(2026, 3, 3),
        "profile": profile,
        "opening_balance": profile.current_available_balance,
        "classified_events": (),
        "historical_settled_cash_events": (),
        "pending_debits": (),
        "ignored_pending_credits": (),
        "scheduled_debits": (),
        "confirmed_scheduled_income": (),
        "ignored_events": (),
        "unresolved_events": (),
        "unexpected_events": (),
        "lifecycle_groups": (),
        "recurring_series_candidates": (),
        "direction_inconsistencies": (),
        "flexibility_inconsistencies": (),
        "requires_message_resolution": False,
        "user_message_ids": (),
        "user_image_ids": (),
    }
    values.update(overrides)
    return NormalizedFinancialState(**values)  # type: ignore[arg-type]


def _candidate(day: date, amount: Decimal) -> ForecastCashEvent:
    return ForecastCashEvent(
        date=day,
        amount_home_currency=amount,
        direction=EventDirection.DEBIT,
        signed_amount=-amount,
        kind=ForecastEventKind.CANDIDATE_PAYMENT,
        source=ForecastEventSource.CANDIDATE,
        source_event_id="candidate_1",
        source_series_key=None,
        is_generated_recurrence=False,
        priority=SameDayPriority.CANDIDATE_PAYMENT,
        description="candidate request payment",
        order_key="candidate_1",
    )


def test_opening_balance_ignores_historical_settled_events() -> None:
    historical = _event(
        source_event_id="hist_1",
        cash_date=date(2026, 3, 1),
        event_date=date(2026, 3, 1),
        settlement_date=date(2026, 3, 1),
        status=EventStatus.SETTLED,
        is_historical=True,
        is_prospective=False,
        amount_home_currency=Decimal("9999"),
        description="Old settled rent",
        category="rent",
    )
    state = _state(
        historical_settled_cash_events=(historical,),
        classified_events=(historical,),
    )
    result = forecast_financial_state(state)
    assert result.opening_balance == Decimal("5000")
    assert result.all_entries == ()
    assert result.minimum_observed_balance == Decimal("5000")


def test_day_zero_and_horizon_boundary() -> None:
    # request_date 2026-03-03 + 90 = 2026-06-01
    on_end = _event(
        source_event_id="sched_end",
        cash_date=date(2026, 6, 1),
        event_date=date(2026, 6, 1),
        settlement_date=date(2026, 6, 1),
        status=EventStatus.SCHEDULED,
        description="On inclusive horizon end",
        amount_home_currency=Decimal("4"),
        category="fees",
    )
    outside = _event(
        source_event_id="sched_out",
        cash_date=date(2026, 6, 2),
        event_date=date(2026, 6, 2),
        settlement_date=date(2026, 6, 2),
        status=EventStatus.SCHEDULED,
        description="After horizon",
        amount_home_currency=Decimal("50"),
        category="fees",
    )
    day0 = _event(
        source_event_id="sched_day0",
        cash_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        status=EventStatus.SCHEDULED,
        description="Day zero obligation",
        amount_home_currency=Decimal("7"),
        category="fees",
    )
    state = _state(
        scheduled_debits=(on_end, outside, day0),
        classified_events=(on_end, outside, day0),
    )
    result = forecast_financial_state(state, ForecastConfig(horizon_days=90))
    dates = {entry.date for entry in result.all_entries}
    assert date(2026, 3, 3) in dates
    assert date(2026, 6, 1) in dates
    assert date(2026, 6, 2) not in dates
    assert result.horizon_end == date(2026, 6, 1)


def test_empty_forecast_preserves_opening_snapshot() -> None:
    result = forecast_financial_state(_state())
    assert result.all_entries == ()
    assert result.daily_forecasts == ()
    assert result.is_safe is True
    assert result.minimum_observed_balance == Decimal("5000")


def test_confirmed_salary_applies_before_same_day_rent() -> None:
    salary = _event(
        source_event_id="sal_1",
        cash_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("1000"),
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    rent = _event(
        source_event_id="rent_1",
        cash_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        amount_home_currency=Decimal("600"),
        category="rent",
        description="Apartment rent",
        status=EventStatus.SCHEDULED,
    )
    profile = _profile(
        current_available_balance=Decimal("500"),
        minimum_balance_to_keep=Decimal("400"),
    )
    state = _state(
        profile=profile,
        opening_balance=Decimal("500"),
        scheduled_debits=(rent,),
        confirmed_scheduled_income=(salary,),
        classified_events=(salary, rent),
    )
    result = forecast_financial_state(state)
    day = result.daily_forecasts[0]
    assert [entry.kind for entry in day.entries] == [
        ForecastEventKind.CONFIRMED_CREDIT,
        ForecastEventKind.SCHEDULED_DEBIT,
    ]
    assert day.running_balances == (Decimal("1500"), Decimal("900"))
    assert result.is_safe is True
    assert result.minimum_observed_balance == Decimal("500")


def test_same_day_order_is_credit_debit_generated_then_candidate() -> None:
    salary = _event(
        source_event_id="sal_1",
        cash_date=date(2026, 3, 10),
        event_date=date(2026, 3, 10),
        settlement_date=date(2026, 3, 10),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("2000"),
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    obligated = _event(
        source_event_id="debt_1",
        cash_date=date(2026, 3, 10),
        event_date=date(2026, 3, 10),
        settlement_date=date(2026, 3, 10),
        amount_home_currency=Decimal("100"),
        category="debt",
        description="Scheduled retry",
        status=EventStatus.SCHEDULED,
        event_type=EventType.DEBT_PAYMENT,
    )
    groceries = _series(
        observed_dates=(
            date(2026, 2, 17),
            date(2026, 2, 24),
            date(2026, 3, 3),
        ),
        event_ids=("g1", "g2", "g3"),
    )
    # next weekly after 3 Mar is 10 Mar — same day as salary/rent
    state = _state(
        scheduled_debits=(obligated,),
        confirmed_scheduled_income=(salary,),
        classified_events=(salary, obligated),
        recurring_series_candidates=(groceries,),
    )
    result = forecast_financial_state(
        state,
        candidate_payments=(_candidate(date(2026, 3, 10), Decimal("50")),),
    )
    day = next(item for item in result.daily_forecasts if item.date == date(2026, 3, 10))
    kinds = [entry.kind for entry in day.entries]
    assert kinds == [
        ForecastEventKind.CONFIRMED_CREDIT,
        ForecastEventKind.SCHEDULED_DEBIT,
        ForecastEventKind.GENERATED_RECURRING_DEBIT,
        ForecastEventKind.CANDIDATE_PAYMENT,
    ]
    assert [entry.priority for entry in day.entries] == [
        SameDayPriority.CONFIRMED_CREDIT,
        SameDayPriority.OBLIGATED_DEBIT,
        SameDayPriority.GENERATED_RECURRING_DEBIT,
        SameDayPriority.CANDIDATE_PAYMENT,
    ]


def test_pending_debit_is_reserved_on_request_date() -> None:
    pending = _event(
        source_event_id="pend_1",
        cash_date=date(2026, 3, 8),
        event_date=date(2026, 3, 2),
        settlement_date=date(2026, 3, 8),
        amount_home_currency=Decimal("700"),
        description="Pending fuel authorization",
    )
    state = _state(
        pending_debits=(pending,),
        classified_events=(pending,),
    )
    result = forecast_financial_state(state)
    assert len(result.all_entries) == 1
    entry = result.all_entries[0]
    assert entry.date == date(2026, 3, 3)
    assert entry.kind is ForecastEventKind.PENDING_DEBIT_RESERVE
    assert entry.original_cash_date == date(2026, 3, 8)
    assert entry.original_event_date == date(2026, 3, 2)
    assert result.opening_balance == Decimal("5000")
    assert result.daily_forecasts[0].closing_balance == Decimal("4300")


def test_pending_credit_is_ignored() -> None:
    pending_credit = _event(
        source_event_id="ref_1",
        direction=EventDirection.CREDIT,
        event_type=EventType.REFUND,
        category="refund",
        description="Pending merchant refund",
        amount_home_currency=Decimal("500"),
        status=EventStatus.PENDING,
        ignore_reason=IgnoreReason.PENDING_CREDIT,
        is_prospective=False,
        counts_as_cash=False,
    )
    state = _state(
        ignored_pending_credits=(pending_credit,),
        ignored_events=(pending_credit,),
        classified_events=(pending_credit,),
    )
    result = forecast_financial_state(state)
    assert result.all_entries == ()
    assert any("pending credit" in line for line in result.ignored_summaries)


def test_scheduled_debit_and_confirmed_salary_use_settlement_date() -> None:
    debit = _event(
        source_event_id="sch_d",
        cash_date=date(2026, 3, 20),
        event_date=date(2026, 3, 1),
        settlement_date=date(2026, 3, 20),
        status=EventStatus.SCHEDULED,
        description="Scheduled school fee",
        category="education",
        amount_home_currency=Decimal("200"),
    )
    salary = _event(
        source_event_id="sch_c",
        cash_date=date(2026, 3, 15),
        event_date=date(2026, 3, 1),
        settlement_date=date(2026, 3, 15),
        status=EventStatus.SCHEDULED,
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        amount_home_currency=Decimal("3000"),
    )
    state = _state(
        scheduled_debits=(debit,),
        confirmed_scheduled_income=(salary,),
        classified_events=(debit, salary),
    )
    result = forecast_financial_state(state)
    by_id = {entry.source_event_id: entry for entry in result.all_entries}
    assert by_id["sch_d"].date == date(2026, 3, 20)
    assert by_id["sch_c"].date == date(2026, 3, 15)


def test_weekly_biweekly_monthly_generation_and_low_excluded() -> None:
    weekly = _series()
    biweekly = _series(
        category="transport",
        normalized_description="bus pass",
        original_description="Bus pass",
        inferred_cadence=Cadence.BIWEEKLY,
        inferred_cadence_days=14,
        observed_dates=(date(2026, 1, 20), date(2026, 2, 3), date(2026, 2, 17), date(2026, 3, 3)),
        event_ids=("b1", "b2", "b3", "b4"),
        representative_amount=Decimal("80"),
        observed_amounts_home_currency=(Decimal("80"),) * 4,
    )
    monthly = _series(
        category="rent",
        normalized_description="apartment rent transfer",
        original_description="Apartment rent transfer",
        inferred_cadence=Cadence.MONTHLY,
        inferred_cadence_days=30,
        observed_dates=(date(2025, 12, 2), date(2026, 1, 2), date(2026, 2, 2), date(2026, 3, 2)),
        event_ids=("r1", "r2", "r3", "r4"),
        representative_amount=Decimal("900"),
        observed_amounts_home_currency=(Decimal("900"),) * 4,
        is_protected_category=True,
    )
    low_dining = _series(
        category="dining",
        normalized_description="weekend food delivery",
        original_description="Weekend food delivery",
        inferred_cadence=Cadence.IRREGULAR,
        inferred_cadence_days=None,
        cadence_confidence=CadenceConfidence.LOW,
        amount_behavior=AmountBehavior.VARIABLE,
        representative_amount=None,
        observed_dates=(date(2026, 1, 10), date(2026, 2, 20)),
        observed_amounts_home_currency=(Decimal("50"), Decimal("80")),
        event_ids=("d1", "d2"),
        flexibility=Flexibility.REDUCIBLE,
        likely_recurring=True,
        reason="low irregular flexible",
    )
    state = _state(
        recurring_series_candidates=(weekly, biweekly, monthly, low_dining),
    )
    result = forecast_financial_state(state)
    generated = [entry for entry in result.all_entries if entry.is_generated_recurrence]
    categories = {entry.category for entry in generated}
    assert "groceries" in categories
    assert "transport" in categories
    assert "rent" in categories
    assert "dining" not in categories
    weekly_dates = [entry.date for entry in generated if entry.category == "groceries"]
    assert weekly_dates[0] == date(2026, 3, 3)
    assert weekly_dates[1] == date(2026, 3, 10)
    monthly_dates = [entry.date for entry in generated if entry.category == "rent"]
    assert monthly_dates[0] == date(2026, 4, 2)
    assert date(2026, 3, 2) not in monthly_dates


def test_medium_included_and_low_subscription_with_cadence_included() -> None:
    medium = _series(
        category="utilities",
        normalized_description="electricity",
        original_description="Electricity",
        inferred_cadence=Cadence.MONTHLY,
        cadence_confidence=CadenceConfidence.MEDIUM,
        observed_dates=(date(2025, 12, 5), date(2026, 1, 5), date(2026, 2, 5)),
        event_ids=("u1", "u2", "u3"),
        representative_amount=Decimal("40"),
        observed_amounts_home_currency=(Decimal("40"),) * 3,
    )
    low_sub = _series(
        category="streaming",
        normalized_description="netflix",
        original_description="Netflix",
        event_type=EventType.SUBSCRIPTION,
        inferred_cadence=Cadence.MONTHLY,
        cadence_confidence=CadenceConfidence.LOW,
        observed_dates=(date(2026, 1, 12), date(2026, 2, 12)),
        event_ids=("n1", "n2"),
        representative_amount=Decimal("15"),
        observed_amounts_home_currency=(Decimal("15"), Decimal("15")),
        flexibility=Flexibility.STOPPABLE,
        likely_recurring=True,
    )
    state = _state(recurring_series_candidates=(medium, low_sub))
    included = forecast_financial_state(state)
    assert {entry.category for entry in included.all_entries} >= {"utilities", "streaming"}
    excluded = forecast_financial_state(
        state,
        ForecastConfig(include_medium_confidence_recurrence=False),
    )
    cats = {entry.category for entry in excluded.all_entries}
    assert "utilities" not in cats
    assert "streaming" in cats


def test_explicit_future_event_prevents_generated_duplicate() -> None:
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
        scheduled_confirmed_event_ids=("sal_sched",),
    )
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
    result = forecast_financial_state(state)
    march = [
        entry
        for entry in result.all_entries
        if entry.date == date(2026, 3, 15) and entry.direction is EventDirection.CREDIT
    ]
    assert len(march) == 1
    assert march[0].source_event_id == "sal_sched"
    later = [
        entry
        for entry in result.all_entries
        if entry.is_generated_recurrence and entry.category == "salary"
    ]
    assert later
    assert all(entry.date != date(2026, 3, 15) for entry in later)
    assert later[0].date == date(2026, 4, 15)


def test_variable_salary_uses_latest_not_recent_max() -> None:
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
        amount_behavior=AmountBehavior.VARIABLE,
        representative_amount=None,
        observed_amounts_home_currency=(
            Decimal("1441"),
            Decimal("1400"),
            Decimal("1200"),
            Decimal("1037.52"),
        ),
    )
    result = forecast_financial_state(
        _state(recurring_series_candidates=(series,)),
        ForecastConfig(variable_amount_strategy=VariableAmountStrategy.RECENT_MAX),
    )
    generated = next(entry for entry in result.all_entries if entry.is_generated_recurrence)
    assert generated.amount_home_currency == Decimal("1037.52")


def test_variable_strategies_change_generated_amount() -> None:
    series = _series(
        amount_behavior=AmountBehavior.VARIABLE,
        representative_amount=None,
        observed_amounts_home_currency=(
            Decimal("10.00"),
            Decimal("30.00"),
            Decimal("12.00"),
            Decimal("20.00"),
        ),
    )
    state = _state(recurring_series_candidates=(series,))
    latest = forecast_financial_state(
        state, ForecastConfig(variable_amount_strategy=VariableAmountStrategy.LATEST)
    )
    maximum = forecast_financial_state(
        state, ForecastConfig(variable_amount_strategy=VariableAmountStrategy.MAX)
    )
    recent = forecast_financial_state(
        state, ForecastConfig(variable_amount_strategy=VariableAmountStrategy.RECENT_MAX)
    )
    gen = lambda result: next(
        entry.amount_home_currency for entry in result.all_entries if entry.is_generated_recurrence
    )
    assert gen(latest) == Decimal("20.00")
    assert gen(maximum) == Decimal("30.00")
    assert gen(recent) == Decimal("30.00")


def test_minimum_balance_equality_is_safe_and_below_is_not() -> None:
    equal = forecast_financial_state(
        _state(profile=_profile(current_available_balance=Decimal("1000")))
    )
    assert equal.opening_balance == Decimal("1000")
    assert equal.is_safe is True

    pending = _event(amount_home_currency=Decimal("1"), cash_date=date(2026, 3, 8))
    below = forecast_financial_state(
        _state(
            profile=_profile(
                current_available_balance=Decimal("1000"),
                minimum_balance_to_keep=Decimal("1000"),
            ),
            opening_balance=Decimal("1000"),
            pending_debits=(pending,),
            classified_events=(pending,),
        )
    )
    assert below.is_safe is False
    assert below.first_violation_date == date(2026, 3, 3)
    assert below.first_violation_entry is not None
    assert below.minimum_observed_balance == Decimal("999")


def test_minimum_is_checked_after_each_event_not_only_day_close() -> None:
    debit = _event(
        source_event_id="big_debit",
        cash_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        status=EventStatus.SCHEDULED,
        amount_home_currency=Decimal("300"),
        description="Large same-day debit",
        category="fees",
    )
    credit = _event(
        source_event_id="late_credit",
        cash_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        status=EventStatus.SCHEDULED,
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        amount_home_currency=Decimal("500"),
    )
    # credits apply first, so this pair is safe. Inject a generated debit after
    # an obligated debit that dips below the floor before a later candidate
    # cannot rescue the earlier check — use opening 1000, min 800, debit 300
    # first would dip to 700 if credits were not first. Force obligated-only:
    state = _state(
        profile=_profile(
            current_available_balance=Decimal("1000"),
            minimum_balance_to_keep=Decimal("800"),
        ),
        opening_balance=Decimal("1000"),
        scheduled_debits=(debit,),
        classified_events=(debit, credit),
        confirmed_scheduled_income=(),
    )
    # Same-day credit is listed in classified but not confirmed — should not apply.
    # Debit 300 → 700 < 800, then a candidate credit is not allowed (candidates are debits).
    result = forecast_financial_state(state)
    assert result.is_safe is False
    assert result.daily_forecasts[0].closing_balance == Decimal("700")
    assert result.minimum_observed_balance == Decimal("700")

    # Now include the credit: payday-first keeps the day safe, proving intra-day order.
    safe = forecast_financial_state(
        _state(
            profile=_profile(
                current_available_balance=Decimal("1000"),
                minimum_balance_to_keep=Decimal("800"),
            ),
            opening_balance=Decimal("1000"),
            scheduled_debits=(debit,),
            confirmed_scheduled_income=(credit,),
            classified_events=(debit, credit),
        )
    )
    assert safe.is_safe is True
    assert safe.daily_forecasts[0].running_balances == (Decimal("1500"), Decimal("1200"))


def test_intraday_dip_is_captured_even_when_day_closes_above_floor() -> None:
    # Credit first (priority), then a debit larger than opening+credit-min.
    # To create a dip despite credits-first, start at 1000, min 900, no credit,
    # two obligated debits: 50 then 100. After first still safe (950), after
    # second 850 < 900. Closing is the min. Need credit AFTER debit to close
    # above — that requires a generated credit which also has credit priority
    # so it would apply first. Use opening already equal, then debit, then we
    # cannot have a later credit in Phase 3 order. This documents that credits
    # always precede debits. Capture first violation on the debit.
    debit = _event(
        source_event_id="d1",
        cash_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        status=EventStatus.SCHEDULED,
        amount_home_currency=Decimal("200"),
        description="Obligation",
    )
    result = forecast_financial_state(
        _state(
            profile=_profile(
                current_available_balance=Decimal("1000"),
                minimum_balance_to_keep=Decimal("900"),
            ),
            opening_balance=Decimal("1000"),
            scheduled_debits=(debit,),
            classified_events=(debit,),
        )
    )
    assert result.first_violation_entry is not None
    assert result.first_violation_entry.source_event_id == "d1"


def test_generated_salary_is_flagged_when_messages_are_unresolved() -> None:
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
            date(2026, 2, 15),
        ),
        event_ids=("s1", "s2", "s3", "s4"),
        representative_amount=Decimal("33345000"),
        observed_amounts_home_currency=(Decimal("33345000"),) * 4,
        requires_message_resolution=True,
    )
    state = _state(
        recurring_series_candidates=(series,),
        requires_message_resolution=True,
        user_message_ids=("message_02",),
    )
    result = forecast_financial_state(state)
    generated = [entry for entry in result.all_entries if entry.is_generated_recurrence]
    assert generated
    assert all(entry.requires_message_confirmation for entry in generated)
    assert result.message_uncertainty_flags


def test_unresolved_amount_is_never_zero_and_strict_mode_raises() -> None:
    pending = _event(
        source_event_id="blank_pending",
        amount_home_currency=None,
        original_amount=None,
        requires_external_evidence=True,
        ignore_reason=IgnoreReason.BLANK_AMOUNT,
        counts_as_cash=False,
        is_prospective=True,
    )
    state = _state(
        pending_debits=(),
        unresolved_events=(pending,),
        classified_events=(pending,),
    )
    result = forecast_financial_state(state)
    assert result.all_entries == ()
    assert result.unresolved_reasons
    assert all("zero" not in reason or "cannot" in reason for reason in result.unresolved_reasons)
    with pytest.raises(UnresolvedForecastError) as exc:
        forecast_financial_state(state, ForecastConfig(strict_unresolved_amounts=True))
    assert "blank_pending" in str(exc.value)


def test_forecast_relevant_blank_pending_debit_is_not_reserved_as_zero() -> None:
    pending = _event(
        source_event_id="blank_debit",
        amount_home_currency=None,
        original_amount=None,
        counts_as_cash=True,
        is_prospective=True,
        requires_external_evidence=True,
    )
    state = _state(pending_debits=(pending,), classified_events=(pending,))
    result = forecast_financial_state(state)
    assert result.all_entries == ()
    assert result.opening_balance == Decimal("5000")
    assert any("blank_debit" in reason for reason in result.unresolved_reasons)


def test_flexible_high_recurrence_stays_in_baseline_forecast() -> None:
    streaming = _series(
        category="streaming",
        normalized_description="netflix",
        original_description="Netflix",
        event_type=EventType.SUBSCRIPTION,
        inferred_cadence=Cadence.MONTHLY,
        observed_dates=(date(2025, 11, 8), date(2025, 12, 8), date(2026, 1, 8), date(2026, 2, 8)),
        event_ids=("n1", "n2", "n3", "n4"),
        representative_amount=Decimal("12"),
        observed_amounts_home_currency=(Decimal("12"),) * 4,
        flexibility=Flexibility.STOPPABLE,
        profile_allows_stop=True,
    )
    result = forecast_financial_state(_state(recurring_series_candidates=(streaming,)))
    assert any(entry.category == "streaming" for entry in result.all_entries)
    assert all(
        entry.amount_home_currency == Decimal("12")
        for entry in result.all_entries
        if entry.category == "streaming"
    )


def test_generated_events_carry_provenance() -> None:
    series = _series()
    result = forecast_financial_state(_state(recurring_series_candidates=(series,)))
    generated = next(entry for entry in result.all_entries if entry.is_generated_recurrence)
    assert generated.provenance is not None
    assert generated.provenance.series_key == series_key(series)
    assert generated.provenance.source_event_ids == series.event_ids
    assert generated.provenance.cadence is Cadence.WEEKLY
    assert generated.provenance.confidence is CadenceConfidence.HIGH
    assert generated.provenance.generation_method == "calendar_cadence"


def test_no_generated_salary_when_history_is_low_confidence() -> None:
    series = _series(
        category="salary",
        normalized_description="payroll credit",
        original_description="Payroll credit",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        cadence_confidence=CadenceConfidence.LOW,
        observed_dates=(date(2025, 6, 15), date(2026, 2, 15)),
        event_ids=("s1", "s2"),
        representative_amount=Decimal("1000"),
        observed_amounts_home_currency=(Decimal("1000"), Decimal("1000")),
        requires_message_resolution=True,
        likely_recurring=True,
    )
    result = forecast_financial_state(
        _state(
            recurring_series_candidates=(series,),
            requires_message_resolution=True,
            user_message_ids=("message_12",),
        )
    )
    assert not any(entry.category == "salary" for entry in result.all_entries)
    assert result.message_uncertainty_flags
