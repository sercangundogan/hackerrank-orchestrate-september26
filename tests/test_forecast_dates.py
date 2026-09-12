from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finance.forecast import (
    add_calendar_months,
    iter_recurrence_dates,
    next_recurrence_date,
    resolve_variable_amount,
)
from finance.forecast_models import ForecastConfig, VariableAmountStrategy
from finance.models import Cadence


def test_monthly_calendar_dates_do_not_drift_from_the_15th() -> None:
    start = date(2026, 1, 15)
    assert add_calendar_months(start, 1, pattern_day=15) == date(2026, 2, 15)
    assert add_calendar_months(start, 2, pattern_day=15) == date(2026, 3, 15)
    cursor = start
    expected = [
        date(2026, 2, 15),
        date(2026, 3, 15),
        date(2026, 4, 15),
    ]
    found = []
    for _ in expected:
        cursor = next_recurrence_date(cursor, Cadence.MONTHLY, pattern_day=15)
        found.append(cursor)
    assert found == expected
    assert date(2026, 2, 14) not in found
    assert date(2026, 3, 16) not in found


def test_month_end_pattern_clamps_then_restores() -> None:
    jan = date(2026, 1, 31)
    feb = add_calendar_months(jan, 1, pattern_day=31)
    mar = add_calendar_months(feb, 1, pattern_day=31)
    apr = add_calendar_months(mar, 1, pattern_day=31)
    assert feb == date(2026, 2, 28)
    assert mar == date(2026, 3, 31)
    assert apr == date(2026, 4, 30)


def test_leap_year_month_end() -> None:
    jan = date(2024, 1, 31)
    feb = add_calendar_months(jan, 1, pattern_day=31)
    assert feb == date(2024, 2, 29)


def test_weekly_and_biweekly_advance_by_fixed_days() -> None:
    assert next_recurrence_date(date(2026, 1, 6), Cadence.WEEKLY) == date(2026, 1, 13)
    assert next_recurrence_date(date(2026, 1, 6), Cadence.BIWEEKLY) == date(2026, 1, 20)


def test_iter_recurrence_stays_inside_inclusive_horizon() -> None:
    dates = iter_recurrence_dates(
        last_observed=date(2026, 1, 15),
        cadence=Cadence.MONTHLY,
        horizon_start=date(2026, 3, 3),
        horizon_end=date(2026, 6, 1),
        pattern_day=15,
    )
    assert dates == (date(2026, 3, 15), date(2026, 4, 15), date(2026, 5, 15))


def test_horizon_includes_request_date_plus_90() -> None:
    config = ForecastConfig(horizon_days=90)
    start = date(2026, 3, 6)
    end = config.horizon_end(start)
    assert end == date(2026, 6, 4)
    dates = iter_recurrence_dates(
        last_observed=date(2026, 2, 15),
        cadence=Cadence.MONTHLY,
        horizon_start=start,
        horizon_end=end,
        pattern_day=15,
    )
    assert date(2026, 6, 15) not in dates
    assert date(2026, 3, 15) in dates


def test_variable_amount_strategies_and_decimal_precision() -> None:
    amounts = (Decimal("10.10"), Decimal("12.25"), Decimal("11.00"), Decimal("9.50"))
    assert resolve_variable_amount(amounts, VariableAmountStrategy.LATEST) == Decimal("9.50")
    assert resolve_variable_amount(amounts, VariableAmountStrategy.MAX) == Decimal("12.25")
    assert resolve_variable_amount(amounts, VariableAmountStrategy.RECENT_MAX) == Decimal(
        "12.25"
    )
    short = (Decimal("4.00"), Decimal("6.00"))
    assert resolve_variable_amount(short, VariableAmountStrategy.RECENT_MAX) == Decimal("6.00")
    assert resolve_variable_amount((None, None), VariableAmountStrategy.MAX) is None


def test_unknown_cadence_cannot_be_projected() -> None:
    with pytest.raises(ValueError):
        next_recurrence_date(date(2026, 1, 1), Cadence.IRREGULAR)
