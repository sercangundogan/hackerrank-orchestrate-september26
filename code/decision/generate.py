"""Produce evaluation DecisionResults and write root output.csv."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ai.client import ModelClient
from config import OUTPUT_CSV, OUTPUT_TEMPLATE_CSV
from data.models import FinanceRequest
from data.repository import DatasetRepository
from decision.capacity import compute_capacity
from decision.engine import decide
from decision.models import DecisionResult
from decision.output import OUTPUT_FIELDNAMES, decision_to_row
from decision.output_validator import OutputValidationError, validate_output_file
from evidence.cache import EvidenceCache
from evidence.models import EvidenceBundle
from evidence.pipeline import resolve_request_state
from finance.essential_spending import forecast_config_from_profiles
from finance.forecast_models import ForecastConfig
from finance.models import NormalizedFinancialState
from usage.tracker import UsageTracker


@dataclass(frozen=True)
class GeneratedDecision:
    request: FinanceRequest
    state: NormalizedFinancialState
    bundle: EvidenceBundle
    result: DecisionResult


def generate_evaluation_decisions(
    repository: DatasetRepository,
    *,
    cache: EvidenceCache,
    client: ModelClient,
    tracker: UsageTracker,
    config: ForecastConfig | None = None,
) -> tuple[GeneratedDecision, ...]:
    strategy = config or forecast_config_from_profiles(
        repository.dataset.profiles, strict_unresolved_amounts=True
    )
    rows: list[GeneratedDecision] = []
    for request in repository.dataset.evaluation_requests:
        _, state, bundle = resolve_request_state(
            repository, request, client=client, cache=cache, tracker=tracker
        )
        options = repository.payment_options_for_request(request.request_id)
        capacity = compute_capacity(state, request, strategy)
        result = decide(state, request, options, strategy, capacity=capacity)
        rows.append(
            GeneratedDecision(request=request, state=state, bundle=bundle, result=result)
        )
    return tuple(rows)


def render_output_csv(decisions: tuple[GeneratedDecision, ...]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(OUTPUT_FIELDNAMES), lineterminator="\n")
    writer.writeheader()
    for item in decisions:
        writer.writerow(decision_to_row(item.result))
    return buffer.getvalue()


def write_output_csv(path: Path, decisions: tuple[GeneratedDecision, ...]) -> None:
    if path.resolve() == OUTPUT_TEMPLATE_CSV.resolve():
        raise OutputValidationError("refusing to overwrite dataset/output.csv template")
    path.write_text(render_output_csv(decisions), encoding="utf-8")


def validate_generated(
    path: Path,
    repository: DatasetRepository,
    decisions: tuple[GeneratedDecision, ...],
    config: ForecastConfig,
) -> None:
    states = {item.request.request_id: item.state for item in decisions}
    results = {item.request.request_id: item.result for item in decisions}
    validate_output_file(path, repository, states, results, config)


def diagnostics(decisions: tuple[GeneratedDecision, ...]) -> dict[str, object]:
    status = Counter(item.result.affordability_status.value for item in decisions)
    method = Counter(item.result.recommended_payment_method.value for item in decisions)
    currency = Counter(item.state.profile.home_currency.value for item in decisions)
    request_type = Counter(item.request.request_type.value for item in decisions)
    method_by_type: dict[str, Counter[str]] = {}
    for item in decisions:
        bucket = method_by_type.setdefault(item.request.request_type.value, Counter())
        bucket[item.result.recommended_payment_method.value] += 1
    spending = sum(1 for item in decisions if item.result.spending_changes_needed)
    unresolved = sum(len(item.bundle.failures) for item in decisions)
    zero_safe = sum(1 for item in decisions if item.result.amount_safe_to_pay == 0)
    return {
        "n": len(decisions),
        "status": dict(status),
        "method": dict(method),
        "currency": dict(currency),
        "request_type": dict(request_type),
        "method_by_type": {key: dict(value) for key, value in method_by_type.items()},
        "spending_change_rows": spending,
        "unresolved_evidence": unresolved,
        "zero_safe_amounts": zero_safe,
    }


def default_output_path() -> Path:
    return OUTPUT_CSV
