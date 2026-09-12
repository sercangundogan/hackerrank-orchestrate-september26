"""Raw domain enums and dataclasses for participant-facing dataset files.

These types represent parsed CSV rows. They do not interpret cash-flow
effects, recurrence, affordability, or unstructured evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path


class Currency(str, Enum):
    INR = "INR"
    EUR = "EUR"
    IDR = "IDR"
    ZAR = "ZAR"
    USD = "USD"


class EventDirection(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"
    NON_CASH = "non_cash"


class EventStatus(str, Enum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class EventType(str, Enum):
    EXPENSE = "expense"
    SUBSCRIPTION = "subscription"
    INCOME = "income"
    DEBT_PAYMENT = "debt_payment"
    REFUND = "refund"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"


class Flexibility(str, Enum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    STOPPABLE = "stoppable"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"


class RequestType(str, Enum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class PaymentOptionMethod(str, Enum):
    """Methods that appear in request_payment_options.csv."""

    FULL_PAYMENT = "full_payment"
    INSTALLMENTS = "installments"


class ConsideredPaymentMethod(str, Enum):
    """Methods a user may list in payment_methods_user_will_consider."""

    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"


class RecommendedPaymentMethod(str, Enum):
    """Allowed values for output recommended_payment_method."""

    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class MessageSourceType(str, Enum):
    EMPLOYER = "employer"
    SERVICE_PROVIDER = "service_provider"
    FINANCIAL_SERVICE = "financial_service"
    BANK = "bank"
    MERCHANT = "merchant"


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: Currency
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[ConsideredPaymentMethod, ...]
    max_installment_months: int | None


@dataclass(frozen=True)
class FinanceRequest:
    request_id: str
    user_id: str
    request_date: date
    request_type: RequestType
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class SampleDecision:
    """Ground-truth output fields from sample_requests.csv.

    These exist for later evaluation only. They must not be used to decide
    evaluation requests.
    """

    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedPaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class SampleFinanceRequest:
    request: FinanceRequest
    decision: SampleDecision

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def user_id(self) -> str:
        return self.request.user_id


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: EventType
    description: str
    category: str
    direction: EventDirection
    amount: Decimal | None
    currency: Currency
    event_date: date
    settlement_date: date | None
    status: EventStatus
    linked_event_id: str | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: Currency
    to_currency: Currency
    rate: Decimal

    @property
    def key(self) -> tuple[date, Currency, Currency]:
        return (self.rate_date, self.from_currency, self.to_currency)


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: PaymentOptionMethod
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: datetime
    source_type: MessageSourceType
    message_text: str


@dataclass(frozen=True)
class ImageReference:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str
    path: Path


@dataclass(frozen=True)
class OutputTemplateRow:
    request_id: str
    amount_safe_to_pay: Decimal | None
    affordability_status: AffordabilityStatus | None
    recommended_payment_method: RecommendedPaymentMethod | None
    payment_plan: str | None
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str | None
    decision_explanation: str | None


@dataclass(frozen=True)
class Dataset:
    """In-memory container for every participant-facing CSV."""

    profiles: tuple[FinancialProfile, ...]
    evaluation_requests: tuple[FinanceRequest, ...]
    sample_requests: tuple[SampleFinanceRequest, ...]
    events: tuple[FinancialEvent, ...]
    exchange_rates: tuple[ExchangeRate, ...]
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageReference, ...]
    output_template_rows: tuple[OutputTemplateRow, ...]
    profiles_by_user_id: dict[str, FinancialProfile]
    evaluation_requests_by_id: dict[str, FinanceRequest]
    sample_requests_by_id: dict[str, SampleFinanceRequest]
    events_by_id: dict[str, FinancialEvent]
    exchange_rates_by_key: dict[tuple[date, Currency, Currency], ExchangeRate]
    payment_options_by_id: dict[str, PaymentOption]
    messages_by_id: dict[str, Message]
    images_by_id: dict[str, ImageReference]
    output_template_by_request_id: dict[str, OutputTemplateRow]
