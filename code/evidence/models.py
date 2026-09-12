"""Typed evidence facts extracted from messages and images.

Facts amend the ledger. They never decide affordability and never override
challenge rules, even when a message contains instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from data.models import Currency


class EvidenceSourceType(str, Enum):
    MESSAGE = "message"
    IMAGE = "image"


class FactType(str, Enum):
    EVENT_AMOUNT = "event_amount"
    SALARY_AMOUNT_CHANGE = "salary_amount_change"
    SALARY_PAYMENT_DATE_CHANGE = "salary_payment_date_change"
    EMPLOYMENT_ENDED = "employment_ended"
    TEMPORARY_SALARY_CHANGE = "temporary_salary_change"
    CONFIRMED_FUTURE_INCOME = "confirmed_future_income"
    UNAPPROVED_INCOME = "unapproved_income"
    RECURRING_EXPENSE_CHANGE = "recurring_expense_change"
    RECURRING_EXPENSE_ADDED = "recurring_expense_added"
    REFUND_STILL_PENDING = "refund_still_pending"
    PAYMENT_RETRY_CONFIRMED = "payment_retry_confirmed"
    IGNORE_CREDIT = "ignore_credit"
    SCAM_OR_UNTRUSTED_PAYMENT_REQUEST = "scam_or_untrusted_payment_request"
    OTHER_RELEVANT_FINANCIAL_FACT = "other_relevant_financial_fact"


class FactStatus(str, Enum):
    ACTIVE = "active"
    UNAPPROVED = "unapproved"
    PENDING = "pending"
    CANCELLED = "cancelled"
    UNTRUSTED = "untrusted"


class ExtractionMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    VLM = "vlm"
    CACHE = "cache"


class EvidenceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class EvidenceFact:
    source_type: EvidenceSourceType
    source_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    fact_type: FactType
    amount: Decimal | None
    currency: Currency | None
    effective_date: date | None
    status: FactStatus
    supersedes_event_id: str | None
    confidence: EvidenceConfidence
    extraction_method: ExtractionMethod
    raw_reference: str
    notes: str
    percent: Decimal | None = None
    category: str | None = None


@dataclass(frozen=True)
class ImageExtraction:
    image_id: str
    related_event_id: str
    amount: Decimal | None
    currency: Currency | None
    confidence: EvidenceConfidence
    extraction_method: ExtractionMethod
    rationale: str
    selected_label: str | None = None
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    request_id: str
    user_id: str
    facts: tuple[EvidenceFact, ...]
    image_extractions: tuple[ImageExtraction, ...]
    llm_source_ids: tuple[str, ...]
    vlm_source_ids: tuple[str, ...]
    failures: tuple[str, ...]
