from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from config import OUTPUT_COLUMNS, OUTPUT_TEMPLATE_CSV
from data.models import AffordabilityStatus, RecommendedPaymentMethod
from decision.generate import write_output_csv
from decision.models import (
    CandidatePlan,
    DecisionResult,
    ExplanationFacts,
    PlanPayment,
)
from decision.output import decision_to_row
from decision.output_validator import OutputValidationError
from decision.plans import format_output_amount


def test_format_output_amount_strips_trailing_zeros() -> None:
    assert format_output_amount(Decimal("0")) == "0"
    assert format_output_amount(Decimal("603.3")) == "603.3"
    assert format_output_amount(Decimal("603.30")) == "603.3"
    assert format_output_amount(Decimal("17229139.2")) == "17229139.2"
    assert format_output_amount(Decimal("87170.56")) == "87170.56"
    assert "e" not in format_output_amount(Decimal("25256")).lower()


def test_decision_row_column_order() -> None:
    result = DecisionResult(
        request_id="request_test",
        amount_safe_to_pay=Decimal("100.50"),
        affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
        recommended_payment_method=RecommendedPaymentMethod.FULL_PAYMENT,
        payment_plan=(PlanPayment(date=date(2026, 3, 3), amount=Decimal("100.50")),),
        earliest_date_for_full_payment=date(2026, 3, 3),
        spending_changes_needed=(),
        explanation_facts=ExplanationFacts(
            safe_amount_today=Decimal("100.50"),
            requested_amount=Decimal("100.50"),
            minimum_balance_to_keep=Decimal("10"),
            home_currency="ZAR",
            upcoming_obligation_summaries=(),
            confirmed_income_date=None,
            installment_total=None,
            installment_fee=None,
            spending_change_summaries=(),
            chosen_method="full_payment",
            chosen_status="affordable_now",
            completes_by_deadline=True,
            baseline_full_safe_today=True,
            text="Pay ZAR 100.5 today.",
        ),
        chosen_candidate=CandidatePlan(
            method=RecommendedPaymentMethod.FULL_PAYMENT,
            affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
            payments=(PlanPayment(date=date(2026, 3, 3), amount=Decimal("100.50")),),
            payment_option_id=None,
            spending_changes=(),
            total_paid=Decimal("100.50"),
            start_date=date(2026, 3, 3),
            completion_date=date(2026, 3, 3),
            completes_by_deadline=True,
            requires_spending_changes=False,
            is_preference_eligible=True,
            is_financially_safe=True,
            validation_failures=(),
        ),
        rejected_candidates=(),
    )
    row = decision_to_row(result)
    assert list(row.keys()) == list(OUTPUT_COLUMNS)
    assert row["amount_safe_to_pay"] == "100.5"
    assert row["earliest_date_for_full_payment"] == "2026-03-03"
    assert row["spending_changes_needed"] == "none"


def test_write_refuses_dataset_template() -> None:
    with pytest.raises(OutputValidationError):
        write_output_csv(OUTPUT_TEMPLATE_CSV, ())


def test_production_output_code_quarantines_samples() -> None:
    for path in Path("code/decision").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "SampleDecision" not in text
        assert "sample_requests" not in text
