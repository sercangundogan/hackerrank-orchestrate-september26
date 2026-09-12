"""Internal capacity result types. Not the final output.csv schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finance.forecast_models import ForecastCashEvent, ForecastResult


@dataclass(frozen=True)
class PaymentSafety:
    payment_date: date
    amount: Decimal
    is_safe: bool
    forecast: ForecastResult
    minimum_observed_balance: Decimal
    first_violation_date: date | None
    first_violation_entry: ForecastCashEvent | None


@dataclass(frozen=True)
class CapacityResult:
    request_id: str
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: date | None
    full_payment_safe_today: bool
    baseline_minimum_balance: Decimal
    safe_payment_forecast_minimum: Decimal
    limiting_date: date | None
    limiting_event: ForecastCashEvent | None
    unresolved_reasons: tuple[str, ...]
    quantum: Decimal
    requested_amount: Decimal
    request_date: date
    unsafe_increment_minimum: Decimal | None
    baseline: ForecastResult
    safe_today_forecast: ForecastResult
