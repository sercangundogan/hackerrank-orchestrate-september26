from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from data.models import EventDirection, EventStatus, EventType, Flexibility
from finance.essential_spending import (
    build_category_profiles,
    eligible_essential_categories,
    infer_common_protected_categories,
)
from finance.forecast import forecast_financial_state
from finance.forecast_models import (
    EssentialSpendStrategy,
    ForecastConfig,
    ForecastEventKind,
)
from tests.test_forecast import _event, _profile, _series, _state


def _hist(
    *,
    event_id: str,
    day: date,
    amount: Decimal,
    category: str,
    description: str,
) -> object:
    return _event(
        source_event_id=event_id,
        cash_date=day,
        event_date=day,
        settlement_date=day,
        status=EventStatus.SETTLED,
        is_historical=True,
        is_prospective=False,
        amount_home_currency=amount,
        original_amount=amount,
        category=category,
        description=description,
        flexibility=Flexibility.FIXED,
        event_type=EventType.EXPENSE,
        direction=EventDirection.DEBIT,
    )


def test_multiple_grocery_descriptions_combine() -> None:
    request_date = date(2026, 3, 3)
    events = []
    for index, desc in enumerate(
        ("Supermarket basket", "Local market purchase", "Grocery delivery", "Fresh food shop")
    ):
        for week in range(4):
            events.append(
                _hist(
                    event_id=f"g{index}_{week}",
                    day=request_date - timedelta(days=7 * week + index + 1),
                    amount=Decimal("40"),
                    category="groceries",
                    description=desc,
                )
            )
    state = _state(
        profile=_profile(expense_categories_to_protect=("groceries", "rent")),
        historical_settled_cash_events=tuple(events),
        classified_events=tuple(events),
    )
    result = forecast_financial_state(state)
    reserves = [
        entry
        for entry in result.all_entries
        if entry.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE
        and entry.category == "groceries"
    ]
    assert reserves
    assert sum((entry.amount_home_currency for entry in reserves), Decimal("0")) > Decimal("0")


def test_explicit_recurring_grocery_is_not_double_counted() -> None:
    request_date = date(2026, 3, 3)
    hist = tuple(
        _hist(
            event_id=f"g{i}",
            day=date(2026, 2, 3) + timedelta(days=7 * i),
            amount=Decimal("30"),
            category="groceries",
            description="Weekly groceries",
        )
        for i in range(4)
    )
    series = _series(
        category="groceries",
        normalized_description="weekly groceries",
        original_description="Weekly groceries",
        observed_dates=tuple(date(2026, 2, 3) + timedelta(days=7 * i) for i in range(4)),
        observed_amounts_home_currency=(Decimal("30"),) * 4,
        representative_amount=Decimal("30"),
        event_ids=tuple(f"g{i}" for i in range(4)),
    )
    state = _state(
        profile=_profile(expense_categories_to_protect=("groceries",)),
        historical_settled_cash_events=hist,
        recurring_series_candidates=(series,),
    )
    profiles = build_category_profiles(
        state,
        ForecastConfig(),
        forecast_financial_state(state, ForecastConfig(essential_spend_enabled=False)).all_entries,
    )
    grocery = next(item for item in profiles if item.category == "groceries")
    assert grocery.explicit_projected_amount > Decimal("0")
    assert grocery.residual_reserve >= Decimal("0")
    assert grocery.residual_reserve < grocery.projected_budget


def test_unprotected_discretionary_shopping_not_reserved() -> None:
    request_date = date(2026, 3, 3)
    events = tuple(
        _hist(
            event_id=f"s{i}",
            day=request_date - timedelta(days=10 * (i + 1)),
            amount=Decimal("80"),
            category="shopping",
            description="Personal shopping",
        )
        for i in range(4)
    )
    state = _state(
        profile=_profile(
            expense_categories_to_protect=("rent",),
            expense_categories_user_is_willing_to_reduce=("shopping",),
        ),
        historical_settled_cash_events=events,
    )
    result = forecast_financial_state(state)
    assert not any(
        entry.category == "shopping" and entry.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE
        for entry in result.all_entries
    )


def test_protected_variable_spending_is_forecast() -> None:
    request_date = date(2026, 3, 3)
    events = tuple(
        _hist(
            event_id=f"t{i}",
            day=request_date - timedelta(days=6 * (i + 1)),
            amount=Decimal("15") + Decimal(i),
            category="transport",
            description=f"Trip {i}",
        )
        for i in range(6)
    )
    state = _state(
        profile=_profile(expense_categories_to_protect=("transport",)),
        historical_settled_cash_events=events,
    )
    result = forecast_financial_state(state)
    assert any(
        entry.category == "transport" and entry.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE
        for entry in result.all_entries
    )


def test_one_huge_one_off_does_not_recur() -> None:
    events = (
        _hist(
            event_id="h1",
            day=date(2026, 2, 1),
            amount=Decimal("9000"),
            category="healthcare",
            description="Hospital bill",
        ),
    )
    state = _state(
        profile=_profile(expense_categories_to_protect=("healthcare",)),
        historical_settled_cash_events=events,
    )
    result = forecast_financial_state(state)
    assert not any(
        entry.category == "healthcare" and entry.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE
        for entry in result.all_entries
    )


def test_income_never_enters_essential_reserve() -> None:
    salary = _event(
        source_event_id="sal",
        cash_date=date(2026, 2, 15),
        event_date=date(2026, 2, 15),
        settlement_date=date(2026, 2, 15),
        status=EventStatus.SETTLED,
        is_historical=True,
        is_prospective=False,
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        category="salary",
        description="Payroll credit",
        amount_home_currency=Decimal("3000"),
    )
    state = _state(
        profile=_profile(expense_categories_to_protect=("salary", "rent")),
        historical_settled_cash_events=(salary,),
    )
    result = forecast_financial_state(state)
    assert not any(
        entry.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE and entry.category == "salary"
        for entry in result.all_entries
    )


def test_only_events_before_request_date_are_used() -> None:
    future = _hist(
        event_id="future",
        day=date(2026, 3, 10),
        amount=Decimal("500"),
        category="groceries",
        description="Future shop",
    )
    past = tuple(
        _hist(
            event_id=f"p{i}",
            day=date(2026, 2, 4) + timedelta(days=7 * i),
            amount=Decimal("20"),
            category="groceries",
            description="Past shop",
        )
        for i in range(4)
    )
    state = _state(
        profile=_profile(expense_categories_to_protect=("groceries",)),
        historical_settled_cash_events=past + (future,),
    )
    profiles = build_category_profiles(state, ForecastConfig(), ())
    grocery = next(item for item in profiles if item.category == "groceries")
    assert "future" not in grocery.historical_event_ids


def test_residual_uses_decimal_and_is_non_negative() -> None:
    events = tuple(
        _hist(
            event_id=f"g{i}",
            day=date(2026, 2, 1) + timedelta(days=8 * i),
            amount=Decimal("12.50"),
            category="groceries",
            description="Shop",
        )
        for i in range(5)
    )
    state = _state(
        profile=_profile(expense_categories_to_protect=("groceries",)),
        historical_settled_cash_events=events,
    )
    profiles = build_category_profiles(state, ForecastConfig(), ())
    grocery = next(item for item in profiles if item.category == "groceries")
    assert isinstance(grocery.residual_reserve, Decimal)
    assert grocery.residual_reserve >= Decimal("0")


def test_existing_recurrence_remains_intact() -> None:
    series = _series()
    state = _state(
        profile=_profile(expense_categories_to_protect=("groceries", "rent")),
        recurring_series_candidates=(series,),
    )
    disabled = forecast_financial_state(state, ForecastConfig(essential_spend_enabled=False))
    enabled = forecast_financial_state(state)
    rentish = [
        entry
        for entry in enabled.all_entries
        if entry.kind is ForecastEventKind.GENERATED_RECURRING_DEBIT
    ]
    assert rentish
    assert {entry.source_event_id for entry in disabled.all_entries}.issubset(
        {entry.source_event_id for entry in enabled.all_entries}
    )


def test_common_protected_categories_come_from_profiles() -> None:
    profiles = (
        _profile(expense_categories_to_protect=("rent", "groceries")),
        _profile(user_id="u2", expense_categories_to_protect=("rent", "transport")),
    )
    # Too few profiles to pass the dataset floor — empty is acceptable.
    inferred = infer_common_protected_categories(profiles + tuple(_profile() for _ in range(20)))
    assert "rent" in inferred
    eligible = eligible_essential_categories(_profile(expense_categories_to_protect=("rent",)), inferred)
    assert "rent" in eligible
