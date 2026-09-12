"""Format a DecisionResult into one output.csv row. No financial policy."""

from __future__ import annotations

from config import OUTPUT_COLUMNS
from decision.models import DecisionResult
from decision.plans import format_earliest, format_output_amount, format_payment_plan, format_spending_changes

OUTPUT_FIELDNAMES = OUTPUT_COLUMNS


def format_explanation(result: DecisionResult) -> str:
    text = " ".join(result.explanation_facts.text.split())
    return text


def decision_to_row(result: DecisionResult) -> dict[str, str]:
    return {
        "request_id": result.request_id,
        "amount_safe_to_pay": format_output_amount(result.amount_safe_to_pay),
        "affordability_status": result.affordability_status.value,
        "recommended_payment_method": result.recommended_payment_method.value,
        "payment_plan": format_payment_plan(result.payment_plan),
        "earliest_date_for_full_payment": format_earliest(result.earliest_date_for_full_payment),
        "spending_changes_needed": format_spending_changes(result.spending_changes_needed),
        "decision_explanation": format_explanation(result),
    }
