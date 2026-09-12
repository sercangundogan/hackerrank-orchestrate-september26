"""Deterministic semantic-consistency checks for image extractions.

Rules are family-based (salary, retail, bill, fare, service). They do not
use image IDs, expected amounts, or model confidence to override a mismatch.
"""

from __future__ import annotations

import re
from decimal import Decimal

from data.models import EventDirection, EventType, FinancialEvent
from evidence.models import (
    EvidenceConfidence,
    ImageExtraction,
    ImageValidation,
    SemanticFamily,
    ValidationStatus,
)

_SALARY_FIELDS = frozenset(
    {
        "net_pay",
        "netpay",
        "salary_amount",
        "net_salary",
        "payable_salary",
        "gross_pay",
        "gross_salary",
        "take_home",
        "take_home_pay",
    }
)
_RETAIL_FIELDS = frozenset(
    {
        "grand_total",
        "grandtotal",
        "total",
        "amount_due",
        "payable_amount",
        "order_total",
        "invoice_total",
        "bill_total",
        "net_amount",
        "invoice_net",
    }
)
_BILL_FIELDS = frozenset(
    {
        "balance_due",
        "amount_due",
        "payable_amount",
        "outstanding_balance",
        "current_due",
        "total_due",
        "amount_payable",
        "net_amount",
    }
)
_FARE_FIELDS = frozenset(
    {
        "fare_total",
        "fare",
        "grand_total",
        "amount_due",
        "total",
        "ride_total",
    }
)
_SERVICE_FIELDS = frozenset(
    {
        "grand_total",
        "total",
        "amount_due",
        "payable_amount",
        "order_total",
        "invoice_total",
        "net_amount",
    }
)
_CASH_FIELDS = frozenset(
    {
        "cash_tendered",
        "cash_received",
        "cash_paid",
        "change",
        "change_due",
        "cash_change",
    }
)
_INVOICE_PARTS = frozenset({"invoice_subtotal", "subtotal", "tax", "tax_only"})

_FAMILY_ALLOWED = {
    SemanticFamily.SALARY: _SALARY_FIELDS,
    SemanticFamily.RETAIL: _RETAIL_FIELDS,
    SemanticFamily.BILL: _BILL_FIELDS,
    SemanticFamily.FARE: _FARE_FIELDS,
    SemanticFamily.SERVICE: _SERVICE_FIELDS,
}

_FAMILY_FORBIDDEN = {
    SemanticFamily.SALARY: _CASH_FIELDS | _INVOICE_PARTS,
    SemanticFamily.RETAIL: _SALARY_FIELDS | _CASH_FIELDS,
    SemanticFamily.BILL: _SALARY_FIELDS | _CASH_FIELDS,
    SemanticFamily.FARE: _SALARY_FIELDS | _CASH_FIELDS,
    SemanticFamily.SERVICE: _SALARY_FIELDS | _CASH_FIELDS,
}


def normalize_field(label: str | None) -> str | None:
    if not label:
        return None
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def infer_semantic_family(event: FinancialEvent) -> SemanticFamily:
    category = (event.category or "").lower()
    description = (event.description or "").lower()
    text = f"{category} {description}"
    if event.event_type is EventType.INCOME or category == "salary" or any(
        token in text for token in ("salary", "payroll", "payslip", "net pay")
    ):
        if event.direction is EventDirection.CREDIT or category == "salary":
            return SemanticFamily.SALARY
    if any(token in text for token in ("taxi", "fare", "ride", "cab", "uber", "airport")):
        return SemanticFamily.FARE
    if category == "transport" and "wallet" not in text:
        return SemanticFamily.FARE
    if any(
        token in text
        for token in (
            "rent",
            "utilities",
            "utility",
            "telecom",
            "water bill",
            "electric",
            "maintenance",
            "outstanding",
            "bill due",
        )
    ) or category in {"rent", "utilities", "housing"}:
        return SemanticFamily.BILL
    if any(
        token in text
        for token in (
            "grocery",
            "groceries",
            "shopping",
            "retail",
            "tote",
            "pantry",
            "supermarket",
        )
    ) or category in {"groceries", "shopping"}:
        return SemanticFamily.RETAIL
    if any(
        token in text
        for token in (
            "hospital",
            "pharmacy",
            "clinic",
            "restaurant",
            "dining",
            "ticket",
            "airline",
            "charging",
        )
    ) or category in {"healthcare", "dining"}:
        return SemanticFamily.SERVICE
    if event.direction is EventDirection.DEBIT:
        return SemanticFamily.RETAIL
    return SemanticFamily.UNKNOWN


def validate_image_semantics(
    event: FinancialEvent,
    extraction: ImageExtraction,
) -> ImageValidation:
    family = infer_semantic_family(event)
    field = normalize_field(extraction.selected_label)
    confidence = extraction.confidence
    if extraction.amount is None:
        return ImageValidation(
            status=ValidationStatus.INVALID,
            validation_reason="no extracted amount",
            semantic_consistency=False,
            extraction_confidence=confidence,
            family=family,
            selected_field=field,
        )
    if field is None:
        return ImageValidation(
            status=ValidationStatus.NEEDS_REVIEW,
            validation_reason="missing semantic field",
            semantic_consistency=False,
            extraction_confidence=confidence,
            family=family,
            selected_field=None,
        )
    if family is SemanticFamily.UNKNOWN:
        return ImageValidation(
            status=ValidationStatus.NEEDS_REVIEW,
            validation_reason="unknown event family; semantic field cannot be confirmed",
            semantic_consistency=False,
            extraction_confidence=confidence,
            family=family,
            selected_field=field,
        )
    forbidden = _FAMILY_FORBIDDEN.get(family, frozenset())
    allowed = _FAMILY_ALLOWED.get(family, frozenset())
    if field in forbidden:
        if field in _SALARY_FIELDS and family is not SemanticFamily.SALARY:
            return ImageValidation(
                status=ValidationStatus.NEEDS_REVIEW,
                validation_reason=(
                    f"{field} is semantically inconsistent with {family.value} events "
                    f"({event.event_type.value}/{event.category})"
                ),
                semantic_consistency=False,
                extraction_confidence=confidence,
                family=family,
                selected_field=field,
            )
        return ImageValidation(
            status=ValidationStatus.INVALID,
            validation_reason=(
                f"{field} is incompatible with {family.value} events "
                f"({event.event_type.value}/{event.category})"
            ),
            semantic_consistency=False,
            extraction_confidence=confidence,
            family=family,
            selected_field=field,
        )
    if field in allowed:
        return ImageValidation(
            status=ValidationStatus.VALID,
            validation_reason=f"{field} is compatible with {family.value} events",
            semantic_consistency=True,
            extraction_confidence=confidence,
            family=family,
            selected_field=field,
        )
    return ImageValidation(
        status=ValidationStatus.NEEDS_REVIEW,
        validation_reason=(
            f"{field} is not a recognized field for {family.value} events "
            f"({event.event_type.value}/{event.category})"
        ),
        semantic_consistency=False,
        extraction_confidence=confidence,
        family=family,
        selected_field=field,
    )


def amounts_equivalent(left: Decimal | None, right: Decimal | None) -> bool:
    """True only when Decimals are exactly equal after Decimal normalization.

    Trailing zeros compare equal (41272.0 == 41272). Distinct fractional
    values do not (8528 != 8528.10). No rounding is applied.
    """
    if not isinstance(left, Decimal) or not isinstance(right, Decimal):
        return False
    return left == right
