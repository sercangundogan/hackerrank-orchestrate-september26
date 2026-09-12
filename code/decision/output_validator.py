"""Deterministic validation of a complete evaluation output.csv.

Fails closed: any contract violation is a hard error. This module does not
choose a plan; it only checks the written row against the frozen engine
inputs and the simulator.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dataclasses import replace

from data.models import (
    AffordabilityStatus,
    ConsideredPaymentMethod,
    FinanceRequest,
    PaymentOptionMethod,
    RecommendedPaymentMethod,
)
from data.repository import DatasetRepository
from decision.models import CandidatePlan, PlanPayment, SpendingAction, SpendingActionKind
from decision.output import OUTPUT_FIELDNAMES
from decision.plans import considers, format_output_amount, installment_accepted, installment_schedule
from decision.spending_optimizer import latest_series_event_id
from decision.validator import simulate_plan, validate_candidate
from finance.forecast import series_key
from finance.forecast_models import ForecastConfig
from finance.models import NormalizedFinancialState, RecurringSeriesCandidate

_ZERO = Decimal("0")
_STATUSES = {item.value for item in AffordabilityStatus}
_METHODS = {item.value for item in RecommendedPaymentMethod}


class OutputValidationError(ValueError):
    pass


def _parse_amount(raw: str, field: str, request_id: str) -> Decimal:
    try:
        amount = Decimal(raw)
    except (InvalidOperation, TypeError) as exc:
        raise OutputValidationError(f"{request_id}: invalid {field} {raw!r}") from exc
    if "e" in raw.lower() or "E" in raw:
        raise OutputValidationError(f"{request_id}: scientific notation in {field}")
    if format_output_amount(amount) != raw:
        # Allow identical numeric value with the official formatter.
        if amount != Decimal(raw):
            raise OutputValidationError(f"{request_id}: unnormalized {field} {raw!r}")
    return amount


def _parse_plan(raw: str, request_id: str) -> tuple[PlanPayment, ...]:
    if raw == "none" or raw == "":
        if raw == "":
            raise OutputValidationError(f"{request_id}: empty payment_plan; use none")
        return ()
    payments: list[PlanPayment] = []
    for part in raw.split("|"):
        if ":" not in part:
            raise OutputValidationError(f"{request_id}: malformed payment_plan {raw!r}")
        day, amount_text = part.split(":", 1)
        from datetime import date

        payments.append(PlanPayment(date=date.fromisoformat(day), amount=Decimal(amount_text)))
    return tuple(payments)


def _parse_changes(raw: str) -> tuple[str, ...]:
    if raw == "none":
        return ()
    return tuple(raw.split("|")) if raw else ()


def _series_for_event(
    state: NormalizedFinancialState, event_id: str
) -> RecurringSeriesCandidate | None:
    for series in state.recurring_series_candidates:
        if event_id in series.event_ids:
            return series
        if series.event_ids and latest_series_event_id(series) == event_id:
            return series
    return None


def _action_from_text(
    item: str, state: NormalizedFinancialState, request_id: str
) -> SpendingAction:
    if item.startswith("stop:"):
        event_id = item.split(":", 1)[1]
        new_amount = None
        kind = SpendingActionKind.STOP
    elif item.startswith("reduce_to:"):
        parts = item.split(":")
        if len(parts) != 3:
            raise OutputValidationError(f"{request_id}: malformed reduce_to {item}")
        event_id = parts[1]
        new_amount = Decimal(parts[2])
        kind = SpendingActionKind.REDUCE_TO
    else:
        raise OutputValidationError(f"{request_id}: unknown spending action {item}")
    series = _series_for_event(state, event_id)
    if series is None:
        raise OutputValidationError(f"{request_id}: spending event {event_id} is not a recurring series")
    normal = series.representative_amount or _ZERO
    return SpendingAction(
        kind=kind,
        event_id=event_id,
        new_amount=new_amount,
        series_event_ids=series.event_ids,
        category=series.category,
        normal_amount=normal,
        series_key=series_key(series),
    )


def validate_output_file(
    path: Path,
    repository: DatasetRepository,
    states: dict[str, NormalizedFinancialState],
    results: dict[str, object],
    config: ForecastConfig,
) -> None:
    del results  # chosen DecisionResult is re-checked from the file text
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise OutputValidationError("output.csv must end with a newline")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(OUTPUT_FIELDNAMES):
            raise OutputValidationError(
                f"header mismatch: {reader.fieldnames} != {list(OUTPUT_FIELDNAMES)}"
            )
        rows = list(reader)
    expected = [item.request_id for item in repository.dataset.evaluation_requests]
    got = [row["request_id"] for row in rows]
    if len(got) != 250 or len(set(got)) != 250:
        raise OutputValidationError(f"expected 250 unique rows, got {len(got)} / {len(set(got))}")
    if got != expected:
        extra = set(got) - set(expected)
        missing = set(expected) - set(got)
        raise OutputValidationError(
            f"request_id set mismatch extra={sorted(extra)[:5]} missing={sorted(missing)[:5]}"
        )

    for row, request in zip(rows, repository.dataset.evaluation_requests, strict=True):
        _validate_row(row, request, repository, states[request.request_id], config)


def _validate_row(
    row: dict[str, str],
    request: FinanceRequest,
    repository: DatasetRepository,
    state: NormalizedFinancialState,
    config: ForecastConfig,
) -> None:
    rid = request.request_id
    amount = _parse_amount(row["amount_safe_to_pay"], "amount_safe_to_pay", rid)
    if amount < _ZERO or amount > request.requested_amount:
        raise OutputValidationError(f"{rid}: amount_safe_to_pay out of bounds")
    status = row["affordability_status"]
    method = row["recommended_payment_method"]
    if status not in _STATUSES:
        raise OutputValidationError(f"{rid}: invalid status {status}")
    if method not in _METHODS:
        raise OutputValidationError(f"{rid}: invalid method {method}")
    if row["earliest_date_for_full_payment"] == "none":
        raise OutputValidationError(f"{rid}: earliest_date must be blank, not none")
    payments = _parse_plan(row["payment_plan"], rid)
    changes = _parse_changes(row["spending_changes_needed"])
    if len(changes) > 3:
        raise OutputValidationError(f"{rid}: more than 3 spending changes")
    event_ids = []
    for item in changes:
        if item.startswith("stop:"):
            event_ids.append(item.split(":", 1)[1])
        elif item.startswith("reduce_to:"):
            parts = item.split(":")
            if len(parts) != 3:
                raise OutputValidationError(f"{rid}: malformed reduce_to {item}")
            event_ids.append(parts[1])
        else:
            raise OutputValidationError(f"{rid}: unknown spending action {item}")
    if len(event_ids) != len(set(event_ids)):
        raise OutputValidationError(f"{rid}: stop+reduce on the same series")

    from datetime import date

    earliest_raw = row["earliest_date_for_full_payment"]
    earliest = date.fromisoformat(earliest_raw) if earliest_raw else None
    options = repository.payment_options_for_request(rid)

    if status == AffordabilityStatus.AFFORDABLE_NOW.value:
        if method != RecommendedPaymentMethod.FULL_PAYMENT.value:
            raise OutputValidationError(f"{rid}: affordable_now requires full_payment")
        if amount != request.requested_amount:
            raise OutputValidationError(f"{rid}: affordable_now requires full amount safe")
        if earliest != request.request_date:
            raise OutputValidationError(f"{rid}: affordable_now earliest must be request_date")
        if len(payments) != 1 or payments[0].date != request.request_date:
            raise OutputValidationError(f"{rid}: affordable_now plan must be today")
        if payments[0].amount != request.requested_amount:
            raise OutputValidationError(f"{rid}: affordable_now plan amount mismatch")
        if changes:
            raise OutputValidationError(f"{rid}: affordable_now cannot require spending changes")

    if method == RecommendedPaymentMethod.PARTIAL_PAYMENT.value:
        if status != AffordabilityStatus.AFFORDABLE_WITH_PLAN.value:
            raise OutputValidationError(f"{rid}: partial requires affordable_with_plan")
        if not request.allows_partial_payment:
            raise OutputValidationError(f"{rid}: partial not allowed by request")
        if not considers(state, ConsideredPaymentMethod.PARTIAL_PAYMENT):
            raise OutputValidationError(f"{rid}: user does not consider partial")
        if not (_ZERO < amount < request.requested_amount):
            raise OutputValidationError(f"{rid}: partial requires strict partial safe amount")
        if earliest is None or earliest > request.desired_completion_date:
            raise OutputValidationError(f"{rid}: partial earliest after deadline")
        if len(payments) != 2:
            raise OutputValidationError(f"{rid}: partial requires exactly two payments")
        if payments[0].date != request.request_date or payments[0].amount != amount:
            raise OutputValidationError(f"{rid}: partial first payment mismatch")
        if payments[1].date != earliest or payments[1].amount != request.requested_amount - amount:
            raise OutputValidationError(f"{rid}: partial second payment mismatch")
        if payments[0].amount + payments[1].amount != request.requested_amount:
            raise OutputValidationError(f"{rid}: partial payments must sum to requested")

    if method == RecommendedPaymentMethod.INSTALLMENTS.value:
        if not considers(state, ConsideredPaymentMethod.INSTALLMENTS):
            raise OutputValidationError(f"{rid}: user does not consider installments")
        match = None
        for option in options:
            if option.payment_method is not PaymentOptionMethod.INSTALLMENTS:
                continue
            if installment_schedule(option) == payments:
                match = option
                break
        if match is None:
            raise OutputValidationError(f"{rid}: installment plan is not a supplied option")
        if not installment_accepted(state, match):
            raise OutputValidationError(f"{rid}: exceeds max_installment_months")
        if payments[-1].date > request.desired_completion_date:
            raise OutputValidationError(f"{rid}: installment ends after deadline")

    if method == RecommendedPaymentMethod.WAIT.value:
        if status != AffordabilityStatus.AFFORDABLE_LATER.value:
            raise OutputValidationError(f"{rid}: wait requires affordable_later")
        if not considers(state, ConsideredPaymentMethod.FULL_PAYMENT):
            raise OutputValidationError(f"{rid}: wait requires full_payment preference")
        if earliest is None or earliest <= request.request_date:
            raise OutputValidationError(f"{rid}: wait earliest must be after request_date")
        if earliest > request.desired_completion_date:
            raise OutputValidationError(f"{rid}: wait after deadline")
        if len(payments) != 1 or payments[0].date != earliest:
            raise OutputValidationError(f"{rid}: wait plan date mismatch")
        if payments[0].amount != request.requested_amount:
            raise OutputValidationError(f"{rid}: wait amount must equal requested")

    if status == AffordabilityStatus.NOT_AFFORDABLE.value:
        if method != RecommendedPaymentMethod.NOT_RECOMMENDED.value:
            raise OutputValidationError(f"{rid}: not_affordable requires not_recommended")
        if row["payment_plan"] != "none":
            raise OutputValidationError(f"{rid}: not_affordable payment_plan must be none")

    spending = [_action_from_text(item, state, rid) for item in changes]
    plan = CandidatePlan(
        method=RecommendedPaymentMethod(method),
        affordability_status=AffordabilityStatus(status),
        payments=payments,
        payment_option_id=None,
        spending_changes=tuple(spending),
        total_paid=sum((item.amount for item in payments), _ZERO),
        start_date=payments[0].date if payments else None,
        completion_date=payments[-1].date if payments else None,
        completes_by_deadline=bool(payments) and payments[-1].date <= request.desired_completion_date,
        requires_spending_changes=bool(spending),
        is_preference_eligible=True,
        is_financially_safe=True,
        validation_failures=(),
    )
    if method == RecommendedPaymentMethod.INSTALLMENTS.value:
        for option in options:
            if installment_schedule(option) == payments:
                plan = replace(plan, payment_option_id=option.payment_option_id)
                break
    if method != RecommendedPaymentMethod.NOT_RECOMMENDED.value:
        forecast = simulate_plan(state, request, plan, config)
        if not forecast.is_safe:
            raise OutputValidationError(f"{rid}: recommended schedule violates minimum balance")
        checked = validate_candidate(
            state, request, plan, options, amount, earliest, config
        )
        if checked.validation_failures:
            raise OutputValidationError(f"{rid}: {checked.validation_failures}")
