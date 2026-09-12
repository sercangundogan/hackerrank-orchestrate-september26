"""Normalized financial-state types for Phase 2.

These models classify raw events. They do not simulate cash flow or decide
affordability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from data.models import (
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinancialProfile,
    Flexibility,
)


class LifecycleType(str, Enum):
    CHARGE_THEN_REFUND = "charge_then_refund"
    AUTHORIZATION_THEN_CAPTURE = "authorization_then_capture"
    PURCHASE_THEN_PENDING_REFUND = "purchase_then_pending_refund"
    FAILED_THEN_RETRY = "failed_then_retry"
    INVESTMENT_PURCHASE_THEN_VALUATION = "investment_purchase_then_valuation"
    INVESTMENT_PURCHASE_THEN_SALE = "investment_purchase_then_sale"
    POSSIBLE_DUPLICATE_PENDING_CHARGE = "possible_duplicate_pending_charge"
    UNKNOWN = "unknown"


class LifecycleRole(str, Enum):
    STANDALONE = "standalone"
    PARENT = "parent"
    CHILD = "child"


class Cadence(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class CadenceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AmountBehavior(str, Enum):
    FIXED = "fixed"
    VARIABLE = "variable"
    UNKNOWN = "unknown"


class IgnoreReason(str, Enum):
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"
    PENDING_CREDIT = "pending_credit"
    BLANK_AMOUNT = "blank_amount"
    MISSING_SETTLEMENT_DATE = "missing_settlement_date"
    MISSING_EXCHANGE_RATE = "missing_exchange_rate"
    UNEXPECTED_SETTLED_ON_OR_AFTER_REQUEST = "unexpected_settled_on_or_after_request"
    UNCONFIRMED_SCHEDULED_CREDIT = "unconfirmed_scheduled_credit"
    DIRECTION_MISMATCH = "direction_mismatch"


class UnresolvedCashAmountError(ValueError):
    """Raised when Phase 3+ tries to consume an unresolved cash amount."""


@dataclass(frozen=True)
class NormalizedCashEvent:
    source_event_id: str
    user_id: str
    cash_date: date | None
    event_date: date
    settlement_date: date | None
    direction: EventDirection
    amount_home_currency: Decimal | None
    original_amount: Decimal | None
    original_currency: Currency
    event_type: EventType
    category: str
    description: str
    status: EventStatus
    lifecycle_role: LifecycleRole
    lifecycle_type: LifecycleType | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None
    is_historical: bool
    is_prospective: bool
    counts_as_cash: bool
    ignore_reason: IgnoreReason | None
    requires_external_evidence: bool

    def consumable_home_amount(self) -> Decimal:
        """Return a cash amount only when it is safe for later simulation."""
        if (
            not self.counts_as_cash
            or self.requires_external_evidence
            or self.amount_home_currency is None
            or self.ignore_reason is not None
        ):
            raise UnresolvedCashAmountError(
                f"event {self.source_event_id} is not a consumable cash amount "
                f"(counts_as_cash={self.counts_as_cash}, "
                f"requires_external_evidence={self.requires_external_evidence}, "
                f"amount_home_currency={self.amount_home_currency}, "
                f"ignore_reason={self.ignore_reason})"
            )
        return self.amount_home_currency


@dataclass(frozen=True)
class EventLifecycle:
    parent_event_id: str
    child_event_id: str
    lifecycle_type: LifecycleType
    cash_effect_summary: str


@dataclass(frozen=True)
class RecurringSeriesCandidate:
    user_id: str
    category: str
    normalized_description: str
    original_description: str
    event_ids: tuple[str, ...]
    direction: EventDirection
    event_type: EventType
    observed_dates: tuple[date, ...]
    observed_amounts_home_currency: tuple[Decimal | None, ...]
    inferred_cadence: Cadence
    inferred_cadence_days: int | None
    cadence_confidence: CadenceConfidence
    amount_behavior: AmountBehavior
    representative_amount: Decimal | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None
    is_protected_category: bool
    profile_allows_reduce: bool
    profile_allows_stop: bool
    likely_recurring: bool
    scheduled_confirmed_event_ids: tuple[str, ...]
    requires_message_resolution: bool
    reason: str


@dataclass(frozen=True)
class NormalizedFinancialState:
    request_id: str
    user_id: str
    request_date: date
    profile: FinancialProfile
    opening_balance: Decimal
    classified_events: tuple[NormalizedCashEvent, ...]
    historical_settled_cash_events: tuple[NormalizedCashEvent, ...]
    pending_debits: tuple[NormalizedCashEvent, ...]
    ignored_pending_credits: tuple[NormalizedCashEvent, ...]
    scheduled_debits: tuple[NormalizedCashEvent, ...]
    confirmed_scheduled_income: tuple[NormalizedCashEvent, ...]
    ignored_events: tuple[NormalizedCashEvent, ...]
    unresolved_events: tuple[NormalizedCashEvent, ...]
    unexpected_events: tuple[NormalizedCashEvent, ...]
    lifecycle_groups: tuple[EventLifecycle, ...]
    recurring_series_candidates: tuple[RecurringSeriesCandidate, ...]
    direction_inconsistencies: tuple[str, ...]
    flexibility_inconsistencies: tuple[str, ...]
    requires_message_resolution: bool
    user_message_ids: tuple[str, ...]
    user_image_ids: tuple[str, ...]

    def prospective_cash_events(self) -> tuple[NormalizedCashEvent, ...]:
        return tuple(
            event
            for event in self.classified_events
            if event.is_prospective and event.counts_as_cash
        )
