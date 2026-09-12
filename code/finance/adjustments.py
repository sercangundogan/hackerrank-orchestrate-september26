"""Forecast adjustments produced by evidence resolution.

These are not payment decisions. They only change which future cash events
the Phase 3 simulator is allowed to generate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum


class ForecastAdjustmentKind(str, Enum):
    SALARY_AMOUNT = "salary_amount"
    SALARY_TEMPORARY = "salary_temporary"
    SALARY_PAYDAY = "salary_payday"
    STOP_SALARY_PROJECTION = "stop_salary_projection"
    START_SALARY = "start_salary"
    RENT_PERCENT = "rent_percent"
    CONFIRM_INCOME = "confirm_income"


@dataclass(frozen=True)
class ForecastAdjustment:
    kind: ForecastAdjustmentKind
    amount_home: Decimal | None = None
    percent: Decimal | None = None
    effective_date: date | None = None
    occurrences: int | None = None
    category: str | None = None
    source_ids: tuple[str, ...] = ()
    notes: str = ""
