"""Typed models for the Phase 3 cash-flow forecast.

These types describe a deterministic 90-day simulation. They do not compute
safe-to-pay amounts, affordability, or payment recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum, IntEnum

from data.models import EventDirection, Flexibility
from finance.models import Cadence, CadenceConfidence


class PendingDebitPolicy(str, Enum):
    """How pending debits enter the forecast.

    IMMEDIATE_RESERVE: keep the opening-balance snapshot unchanged and emit an
    obligated debit on request_date so already-authorized outflows reduce cash
    immediately. Original event dates stay on the entry for diagnostics.
    """

    IMMEDIATE_RESERVE = "immediate_reserve"


class VariableAmountStrategy(str, Enum):
    """How a variable recurring series chooses its projected amount."""

    LATEST = "latest"
    MAX = "max"
    RECENT_MAX = "recent_max"


class EssentialSpendStrategy(str, Enum):
    """How category-level residual essential spend is projected."""

    DAILY_RATE = "daily_rate"
    WEEKLY_MEDIAN = "weekly_median"
    WEEKLY_MAX = "weekly_max"
    MONTHLY_MAX = "monthly_max"
    HYBRID = "hybrid"


class SalaryProjectionMode(str, Enum):
    """How future salary occurrences are chosen.

    SCHEDULED_PLUS_REGULAR_HISTORY: apply every confirmed scheduled salary,
    then generate later paydays only from HIGH/MEDIUM historical series.
    Unresolved salary/employment messages are flagged, not interpreted.
    """

    SCHEDULED_PLUS_REGULAR_HISTORY = "scheduled_plus_regular_history"


class SameDayPriority(IntEnum):
    """Deterministic intra-day application order.

    Lower values apply first. Candidate request payments are reserved for
    Phase 4 and are never generated here.
    """

    CONFIRMED_CREDIT = 10
    OBLIGATED_DEBIT = 20
    GENERATED_RECURRING_DEBIT = 30
    CANDIDATE_PAYMENT = 40


class ForecastEventKind(str, Enum):
    CONFIRMED_CREDIT = "confirmed_credit"
    PENDING_DEBIT_RESERVE = "pending_debit_reserve"
    SCHEDULED_DEBIT = "scheduled_debit"
    GENERATED_RECURRING_CREDIT = "generated_recurring_credit"
    GENERATED_RECURRING_DEBIT = "generated_recurring_debit"
    ESSENTIAL_SPEND_RESERVE = "essential_spend_reserve"
    CANDIDATE_PAYMENT = "candidate_payment"


class ForecastEventSource(str, Enum):
    EXPLICIT_EVENT = "explicit_event"
    RESERVED_PENDING = "reserved_pending"
    GENERATED_RECURRENCE = "generated_recurrence"
    CATEGORY_RESERVE = "category_reserve"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class ForecastConfig:
    """Small set of forecast assumptions that later phases can vary."""

    horizon_days: int = 90
    pending_debit_policy: PendingDebitPolicy = PendingDebitPolicy.IMMEDIATE_RESERVE
    variable_amount_strategy: VariableAmountStrategy = VariableAmountStrategy.RECENT_MAX
    include_medium_confidence_recurrence: bool = True
    strict_unresolved_amounts: bool = False
    recent_max_window: int = 3
    salary_projection_mode: SalaryProjectionMode = (
        SalaryProjectionMode.SCHEDULED_PLUS_REGULAR_HISTORY
    )
    essential_spend_enabled: bool = True
    essential_spend_strategy: EssentialSpendStrategy = EssentialSpendStrategy.DAILY_RATE
    essential_lookback_days: int = 90
    extra_essential_categories: frozenset[str] = frozenset()

    def horizon_end(self, request_date: date) -> date:
        """Inclusive end date: request_date through request_date + horizon_days."""
        from datetime import timedelta

        return request_date + timedelta(days=self.horizon_days)


@dataclass(frozen=True)
class RecurrenceProvenance:
    series_key: str
    source_event_ids: tuple[str, ...]
    generation_method: str
    cadence: Cadence
    amount_strategy: str
    confidence: CadenceConfidence
    requires_message_confirmation: bool = False


@dataclass(frozen=True)
class ForecastCashEvent:
    date: date
    amount_home_currency: Decimal
    direction: EventDirection
    signed_amount: Decimal
    kind: ForecastEventKind
    source: ForecastEventSource
    source_event_id: str | None
    source_series_key: str | None
    is_generated_recurrence: bool
    priority: SameDayPriority
    description: str
    original_event_date: date | None = None
    original_settlement_date: date | None = None
    original_cash_date: date | None = None
    provenance: RecurrenceProvenance | None = None
    flexibility: Flexibility | None = None
    category: str | None = None
    requires_message_confirmation: bool = False
    order_key: str = ""


@dataclass(frozen=True)
class DailyForecast:
    date: date
    opening_balance: Decimal
    inflows: Decimal
    outflows: Decimal
    closing_balance: Decimal
    minimum_balance: Decimal
    violated_minimum: bool
    entries: tuple[ForecastCashEvent, ...]
    running_balances: tuple[Decimal, ...]


@dataclass(frozen=True)
class ForecastResult:
    request_id: str
    opening_balance: Decimal
    minimum_balance_to_keep: Decimal
    horizon_start: date
    horizon_end: date
    daily_forecasts: tuple[DailyForecast, ...]
    all_entries: tuple[ForecastCashEvent, ...]
    minimum_observed_balance: Decimal
    first_violation_date: date | None
    first_violation_entry: ForecastCashEvent | None
    is_safe: bool
    unresolved_reasons: tuple[str, ...]
    generated_recurrence_summaries: tuple[str, ...]
    ignored_summaries: tuple[str, ...]
    message_uncertainty_flags: tuple[str, ...]


class UnresolvedForecastError(ValueError):
    """Raised in strict mode when a forecast-relevant amount is unresolved."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        joined = "; ".join(reasons) if reasons else "unresolved forecast amounts"
        super().__init__(joined)
