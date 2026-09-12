"""Sample-label evaluation for Phase 5A capacity fields only.

This module is evaluation, not prediction. Capacity code must not import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ai.client import ModelClient
from data.models import SampleFinanceRequest
from data.repository import DatasetRepository
from decision.capacity import compute_capacity
from evidence.cache import EvidenceCache
from evidence.pipeline import resolve_request_state
from finance.essential_spending import forecast_config_from_profiles
from finance.forecast_models import EssentialSpendStrategy, ForecastConfig, VariableAmountStrategy


@dataclass(frozen=True)
class SampleCapacityRow:
    request_id: str
    requested_amount: Decimal
    predicted_amount: Decimal
    labeled_amount: Decimal
    amount_error: Decimal
    predicted_earliest: date | None
    labeled_earliest: date | None
    amount_match: bool
    date_match: bool
    spending_changes_needed: str
    likely_cause: str


def _classify_mismatch(sample: SampleFinanceRequest, row_notes: str) -> str:
    text = " ".join(
        (
            sample.decision.decision_explanation,
            sample.decision.spending_changes_needed,
            row_notes,
        )
    ).lower()
    if sample.decision.spending_changes_needed != "none":
        return "spending_changes_in_label_not_applied_in_5a"
    if "recurr" in text or "weekly" in text or "monthly" in text:
        return "recurrence"
    if "variable" in text:
        return "variable_spending"
    if "message" in text or "salary" in text or "employ" in text:
        return "message_evidence"
    if "pending" in text:
        return "pending_reserve"
    if "fx" in text or "exchange" in text:
        return "FX"
    if "image" in text or "receipt" in text or "invoice" in text:
        return "image_extraction"
    if "same-day" in text or "payday" in text:
        return "same-day_ordering"
    if row_notes:
        return row_notes
    return "unclassified_or_capacity_search"


def evaluate_sample_capacity(
    repository: DatasetRepository,
    config: ForecastConfig | None = None,
) -> tuple[SampleCapacityRow, ...]:
    strategy = config or forecast_config_from_profiles(
        repository.dataset.profiles,
        strict_unresolved_amounts=True,
    )
    cache = EvidenceCache()
    client = ModelClient(api_key="unused")
    rows: list[SampleCapacityRow] = []
    for sample in repository.dataset.sample_requests:
        request = sample.request
        _, state, _ = resolve_request_state(
            repository, request, client=client, cache=cache
        )
        result = compute_capacity(state, request, strategy)
        amount_error = result.amount_safe_to_pay - sample.decision.amount_safe_to_pay
        amount_match = result.amount_safe_to_pay == sample.decision.amount_safe_to_pay
        date_match = (
            result.earliest_date_for_full_payment
            == sample.decision.earliest_date_for_full_payment
        )
        notes = ""
        if not amount_match or not date_match:
            notes = "field mismatch vs sample label"
        rows.append(
            SampleCapacityRow(
                request_id=request.request_id,
                requested_amount=request.requested_amount,
                predicted_amount=result.amount_safe_to_pay,
                labeled_amount=sample.decision.amount_safe_to_pay,
                amount_error=amount_error,
                predicted_earliest=result.earliest_date_for_full_payment,
                labeled_earliest=sample.decision.earliest_date_for_full_payment,
                amount_match=amount_match,
                date_match=date_match,
                spending_changes_needed=sample.decision.spending_changes_needed,
                likely_cause=""
                if amount_match and date_match
                else _classify_mismatch(sample, notes),
            )
        )
    return tuple(rows)


def summarize(rows: tuple[SampleCapacityRow, ...]) -> dict[str, object]:
    n = len(rows)
    amount_hits = sum(1 for row in rows if row.amount_match)
    date_hits = sum(1 for row in rows if row.date_match)
    mae = sum((abs(row.amount_error) for row in rows), Decimal("0")) / n
    return {
        "n": n,
        "amount_exact": amount_hits,
        "amount_exact_pct": f"{100 * amount_hits / n:.1f}%",
        "mae": mae,
        "earliest_exact": date_hits,
        "earliest_exact_pct": f"{100 * date_hits / n:.1f}%",
        "mismatches": tuple(
            row.request_id for row in rows if not row.amount_match or not row.date_match
        ),
    }


STRATEGY_GRID = (
    ForecastConfig(
        variable_amount_strategy=VariableAmountStrategy.RECENT_MAX,
        include_medium_confidence_recurrence=True,
        strict_unresolved_amounts=True,
    ),
    ForecastConfig(
        variable_amount_strategy=VariableAmountStrategy.LATEST,
        include_medium_confidence_recurrence=True,
        strict_unresolved_amounts=True,
    ),
    ForecastConfig(
        variable_amount_strategy=VariableAmountStrategy.MAX,
        include_medium_confidence_recurrence=True,
        strict_unresolved_amounts=True,
    ),
    ForecastConfig(
        variable_amount_strategy=VariableAmountStrategy.RECENT_MAX,
        include_medium_confidence_recurrence=False,
        strict_unresolved_amounts=True,
    ),
)
