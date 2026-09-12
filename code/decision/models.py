"""Internal capacity and decision result types. Not the final output.csv schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from data.models import AffordabilityStatus, RecommendedPaymentMethod
from finance.forecast_models import ForecastCashEvent, ForecastResult

# Official ranking key 6 uses the lowest payment_option_id. Plans that are
# not seller options use this sentinel so they only lose that last tie-break.
RANKING_OPTION_ID_SENTINEL = "\uffff"


class SpendingActionKind(str, Enum):
    STOP = "stop"
    REDUCE_TO = "reduce_to"


@dataclass(frozen=True)
class PlanPayment:
    date: date
    amount: Decimal


@dataclass(frozen=True)
class SpendingAction:
    kind: SpendingActionKind
    event_id: str
    new_amount: Decimal | None
    series_event_ids: tuple[str, ...]
    category: str
    normal_amount: Decimal
    series_key: str


@dataclass(frozen=True)
class CandidatePlan:
    method: RecommendedPaymentMethod
    affordability_status: AffordabilityStatus
    payments: tuple[PlanPayment, ...]
    payment_option_id: str | None
    spending_changes: tuple[SpendingAction, ...]
    total_paid: Decimal
    start_date: date | None
    completion_date: date | None
    completes_by_deadline: bool
    requires_spending_changes: bool
    is_preference_eligible: bool
    is_financially_safe: bool
    validation_failures: tuple[str, ...]
    forecast_result: ForecastResult | None = None


@dataclass(frozen=True)
class ExplanationFacts:
    safe_amount_today: Decimal
    requested_amount: Decimal
    minimum_balance_to_keep: Decimal
    home_currency: str
    upcoming_obligation_summaries: tuple[str, ...]
    confirmed_income_date: date | None
    installment_total: Decimal | None
    installment_fee: Decimal | None
    spending_change_summaries: tuple[str, ...]
    chosen_method: str
    chosen_status: str
    completes_by_deadline: bool
    baseline_full_safe_today: bool
    text: str


@dataclass(frozen=True)
class DecisionResult:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payment_plan: tuple[PlanPayment, ...]
    earliest_date_for_full_payment: date | None
    spending_changes_needed: tuple[SpendingAction, ...]
    explanation_facts: ExplanationFacts
    chosen_candidate: CandidatePlan
    rejected_candidates: tuple[CandidatePlan, ...]


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
