"""Sample-label evaluation for Phase 5B decision fields.

Evaluation only. Production decision code must not import this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from time import perf_counter

from ai.client import ModelClient
from data.repository import DatasetRepository
from decision.capacity import compute_capacity
from decision.engine import decide
from decision.plans import format_payment_plan, format_spending_changes
from decision.spending_optimizer import legal_actions
from decision.validator import is_recommendable
from evidence.cache import EvidenceCache
from evidence.pipeline import resolve_request_state
from finance.essential_spending import forecast_config_from_profiles


@dataclass(frozen=True)
class SampleDecisionRow:
    request_id: str
    requested_amount: Decimal
    predicted_amount: Decimal
    labeled_amount: Decimal
    amount_match: bool
    amount_error: Decimal
    predicted_status: str
    labeled_status: str
    status_match: bool
    predicted_method: str
    labeled_method: str
    method_match: bool
    predicted_plan: str
    labeled_plan: str
    plan_match: bool
    predicted_earliest: date | None
    labeled_earliest: date | None
    earliest_match: bool
    predicted_changes: str
    labeled_changes: str
    changes_match: bool
    row_match: bool
    verifier_failed: bool
    cause: str
    actions_considered: tuple[str, ...]
    winning_actions: str
    seconds: float


def _normalized_amount_error(predicted: Decimal, labeled: Decimal, requested: Decimal) -> Decimal:
    if requested == 0:
        return Decimal("0")
    return abs(predicted - labeled) / requested


def _classify(row_notes: dict[str, object]) -> str:
    amount_ok = bool(row_notes["amount_match"])
    method_ok = bool(row_notes["method_match"])
    status_ok = bool(row_notes["status_match"])
    plan_ok = bool(row_notes["plan_match"])
    earliest_ok = bool(row_notes["earliest_match"])
    changes_ok = bool(row_notes["changes_match"])
    if all((amount_ok, method_ok, status_ok, plan_ok, earliest_ok, changes_ok)):
        return ""
    if row_notes["verifier_failed"]:
        return "verifier"
    if not amount_ok and method_ok and status_ok and plan_ok:
        return "upstream_capacity_mismatch"
    if not earliest_ok and method_ok:
        return "upstream_capacity_mismatch"
    if not changes_ok and method_ok:
        return "spending_optimizer"
    if not changes_ok:
        return "spending_optimizer"
    if not method_ok and row_notes["labeled_method"] == "installments":
        return "installment_eligibility"
    if not method_ok and row_notes["labeled_method"] == "partial_payment":
        return "candidate_missing"
    if not method_ok and row_notes["labeled_method"] == "wait":
        return "upstream_capacity_mismatch"
    if not plan_ok and method_ok:
        return "formatting"
    if not method_ok:
        return "ranking"
    return "forecast_mismatch"


def evaluate_sample_decisions(repository: DatasetRepository) -> tuple[SampleDecisionRow, ...]:
    config = forecast_config_from_profiles(
        repository.dataset.profiles, strict_unresolved_amounts=True
    )
    cache = EvidenceCache()
    client = ModelClient(api_key="unused")
    rows: list[SampleDecisionRow] = []
    for sample in repository.dataset.sample_requests:
        started = perf_counter()
        request = sample.request
        _, state, _ = resolve_request_state(repository, request, client=client, cache=cache)
        options = repository.payment_options_for_request(request.request_id)
        capacity = compute_capacity(state, request, config)
        result = decide(state, request, options, config, capacity=capacity)
        predicted_plan = format_payment_plan(result.payment_plan)
        predicted_changes = format_spending_changes(result.spending_changes_needed)
        labeled = sample.decision
        amount_match = result.amount_safe_to_pay == labeled.amount_safe_to_pay
        status_match = result.affordability_status is labeled.affordability_status
        method_match = result.recommended_payment_method is labeled.recommended_payment_method
        plan_match = predicted_plan == labeled.payment_plan
        earliest_match = (
            result.earliest_date_for_full_payment == labeled.earliest_date_for_full_payment
        )
        changes_match = predicted_changes == labeled.spending_changes_needed
        row_match = all(
            (
                amount_match,
                status_match,
                method_match,
                plan_match,
                earliest_match,
                changes_match,
            )
        )
        chosen = result.chosen_candidate
        verifier_failed = bool(chosen.validation_failures) and is_recommendable(chosen)
        considered = legal_actions(state, capacity.baseline)
        notes = {
            "amount_match": amount_match,
            "method_match": method_match,
            "status_match": status_match,
            "plan_match": plan_match,
            "earliest_match": earliest_match,
            "changes_match": changes_match,
            "verifier_failed": verifier_failed,
            "labeled_method": labeled.recommended_payment_method.value,
        }
        rows.append(
            SampleDecisionRow(
                request_id=request.request_id,
                requested_amount=request.requested_amount,
                predicted_amount=result.amount_safe_to_pay,
                labeled_amount=labeled.amount_safe_to_pay,
                amount_match=amount_match,
                amount_error=result.amount_safe_to_pay - labeled.amount_safe_to_pay,
                predicted_status=result.affordability_status.value,
                labeled_status=labeled.affordability_status.value,
                status_match=status_match,
                predicted_method=result.recommended_payment_method.value,
                labeled_method=labeled.recommended_payment_method.value,
                method_match=method_match,
                predicted_plan=predicted_plan,
                labeled_plan=labeled.payment_plan,
                plan_match=plan_match,
                predicted_earliest=result.earliest_date_for_full_payment,
                labeled_earliest=labeled.earliest_date_for_full_payment,
                earliest_match=earliest_match,
                predicted_changes=predicted_changes,
                labeled_changes=labeled.spending_changes_needed,
                changes_match=changes_match,
                row_match=row_match,
                verifier_failed=verifier_failed,
                cause=_classify(notes),
                actions_considered=tuple(
                    f"{item.kind.value}:{item.event_id}" for item in considered
                ),
                winning_actions=predicted_changes,
                seconds=perf_counter() - started,
            )
        )
    return tuple(rows)


def summarize(rows: tuple[SampleDecisionRow, ...]) -> dict[str, object]:
    n = len(rows)
    requested = {
        row.request_id: abs(row.amount_error) for row in rows
    }
    amount_hits = sum(1 for row in rows if row.amount_match)
    status_hits = sum(1 for row in rows if row.status_match)
    method_hits = sum(1 for row in rows if row.method_match)
    plan_hits = sum(1 for row in rows if row.plan_match)
    earliest_hits = sum(1 for row in rows if row.earliest_match)
    change_hits = sum(1 for row in rows if row.changes_match)
    row_hits = sum(1 for row in rows if row.row_match)
    verifier_failures = sum(1 for row in rows if row.verifier_failed)
    norms = tuple(_normalized_amount_error(row.predicted_amount, row.labeled_amount, row.requested_amount) for row in rows)
    mae = sum((abs(row.amount_error) for row in rows), Decimal("0")) / n
    median_norm = sorted(norms)[n // 2]
    within_5 = sum(1 for item in norms if item <= Decimal("0.05"))
    return {
        "n": n,
        "amount_exact": amount_hits,
        "amount_mae": mae,
        "amount_median_norm": median_norm,
        "amount_within_5pct": within_5,
        "status_exact": status_hits,
        "method_exact": method_hits,
        "plan_exact": plan_hits,
        "earliest_exact": earliest_hits,
        "changes_exact": change_hits,
        "row_exact": row_hits,
        "verifier_failures": verifier_failures,
        "status_confusion": dict(
            Counter((row.labeled_status, row.predicted_status) for row in rows if not row.status_match)
        ),
        "method_confusion": dict(
            Counter((row.labeled_method, row.predicted_method) for row in rows if not row.method_match)
        ),
        "causes": dict(Counter(row.cause for row in rows if row.cause)),
        "runtime_s": sum(row.seconds for row in rows),
        "abs_errors": requested,
    }
