"""Analysis-only harness for reconstructing the organizer's variable-spend rule.

Not imported by production capacity or forecast defaults.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from ai.client import ModelClient
from data.models import EventDirection, EventType
from data.repository import DatasetRepository
from decision.capacity import compute_capacity, is_payment_safe
from evidence.cache import EvidenceCache
from evidence.pipeline import resolve_request_state
from finance.essential_spending import (
    eligible_essential_categories,
    forecast_config_from_profiles,
)
from finance.forecast_models import (
    EssentialSpendStrategy,
    ForecastConfig,
    ForecastEventKind,
)
from finance.models import NormalizedFinancialState


@dataclass(frozen=True)
class ImpliedGapRow:
    request_id: str
    predicted: Decimal
    labeled: Decimal
    delta: Decimal
    floor: Decimal
    label_min: Decimal
    implied_forecast_gap: Decimal
    limiting_date: date | None
    limiting_event: str
    is_label_safe: bool
    near_floor: bool
    quantum: Decimal


def _day(event) -> date | None:
    return event.cash_date or event.event_date


def _variable_categories(state: NormalizedFinancialState, extras: frozenset[str]) -> frozenset[str]:
    eligible = eligible_essential_categories(state.profile, extras)
    contractual = {
        "rent",
        "debt_repayment",
        "insurance",
        "education",
        "housing",
        "family_support",
    }
    return frozenset(
        category
        for category in eligible
        if category not in contractual
        or category in {"groceries", "transport", "utilities", "healthcare"}
    )


def load_resolved_samples(repository: DatasetRepository):
    cache = EvidenceCache()
    client = ModelClient(api_key="unused")
    rows = []
    for sample in repository.dataset.sample_requests:
        _, state, bundle = resolve_request_state(
            repository, sample.request, client=client, cache=cache
        )
        rows.append((sample, state, bundle))
    return rows


def implied_gaps(repository: DatasetRepository, config: ForecastConfig) -> tuple[ImpliedGapRow, ...]:
    out: list[ImpliedGapRow] = []
    for sample, state, _bundle in load_resolved_samples(repository):
        request = sample.request
        cap = compute_capacity(state, request, config)
        label = sample.decision.amount_safe_to_pay
        paid = is_payment_safe(state, request, request.request_date, label, config)
        floor = state.profile.minimum_balance_to_keep
        gap = paid.minimum_observed_balance - floor
        limiting = paid.forecast.first_violation_entry or cap.limiting_event
        quantum = cap.quantum
        out.append(
            ImpliedGapRow(
                request_id=request.request_id,
                predicted=cap.amount_safe_to_pay,
                labeled=label,
                delta=cap.amount_safe_to_pay - label,
                floor=floor,
                label_min=paid.minimum_observed_balance,
                implied_forecast_gap=gap,
                limiting_date=paid.first_violation_date or cap.limiting_date,
                limiting_event=(
                    f"{limiting.category}:{limiting.description}" if limiting is not None else ""
                ),
                is_label_safe=paid.is_safe,
                near_floor=abs(gap) <= quantum,
                quantum=quantum,
            )
        )
    return tuple(out)


def category_history_stats(state: NormalizedFinancialState, category: str, request_date: date):
    events = []
    for event in state.historical_settled_cash_events:
        day = _day(event)
        if day is None or day >= request_date:
            continue
        if event.direction is not EventDirection.DEBIT:
            continue
        if event.category != category:
            continue
        if event.amount_home_currency is None:
            continue
        if event.event_type is EventType.INCOME:
            continue
        events.append((day, event.amount_home_currency, event.description))
    events.sort()
    if len(events) < 2:
        return None
    gaps = [(events[i][0] - events[i - 1][0]).days for i in range(1, len(events))]
    amounts = [item[1] for item in events]
    start = request_date - timedelta(days=90)
    last90 = [item for item in events if item[0] >= start]
    weekly: dict[date, Decimal] = defaultdict(lambda: Decimal("0"))
    monthly: dict[tuple[int, int], Decimal] = defaultdict(lambda: Decimal("0"))
    for day, amount, _desc in last90:
        week = day - timedelta(days=day.weekday())
        weekly[week] += amount
        monthly[(day.year, day.month)] += amount
    return {
        "n": len(events),
        "n90": len(last90),
        "descs": len({item[2] for item in events}),
        "median_gap": Decimal(median(gaps)) if gaps else Decimal("0"),
        "median_amt": Decimal(str(median(amounts))),
        "mean_amt": sum(amounts, Decimal("0")) / Decimal(len(amounts)),
        "hist90": sum((item[1] for item in last90), Decimal("0")),
        "weekly_median": Decimal(str(median(weekly.values()))) if weekly else Decimal("0"),
        "weekly_max": max(weekly.values()) if weekly else Decimal("0"),
        "monthly_mean": (
            sum(monthly.values(), Decimal("0")) / Decimal(len(monthly)) if monthly else Decimal("0")
        ),
        "monthly_max": max(monthly.values()) if monthly else Decimal("0"),
        "cadence_90": (
            Decimal(str(median(amounts))) * (Decimal("90") / Decimal(median(gaps)))
            if gaps and median(gaps) > 0
            else Decimal("0")
        ),
    }


def salary_audit(state: NormalizedFinancialState, forecast) -> dict[str, object]:
    generated = [
        entry
        for entry in forecast.all_entries
        if entry.signed_amount > 0 and entry.is_generated_recurrence
    ]
    explicit = [
        entry
        for entry in forecast.all_entries
        if entry.signed_amount > 0 and not entry.is_generated_recurrence
    ]
    other = [entry for entry in generated if entry.category != "salary"]
    return {
        "explicit_n": len(explicit),
        "explicit_amt": sum((e.amount_home_currency for e in explicit), Decimal("0")),
        "gen_n": len(generated),
        "gen_amt": sum((e.amount_home_currency for e in generated), Decimal("0")),
        "non_salary_credits": tuple(
            f"{e.category}:{e.description}:{e.amount_home_currency}" for e in other
        ),
        "adjustments": tuple(item.kind.value for item in state.forecast_adjustments),
    }


def paycycle_totals(state: NormalizedFinancialState, category: str, request_date: date):
    salaries = []
    for event in state.historical_settled_cash_events:
        day = _day(event)
        if day is None or day >= request_date:
            continue
        if event.category == "salary" and event.direction is EventDirection.CREDIT:
            salaries.append(day)
    salaries = sorted(set(salaries))
    if len(salaries) < 2:
        return []
    cycles = []
    for start, end in zip(salaries, salaries[1:]):
        total = Decimal("0")
        for event in state.historical_settled_cash_events:
            day = _day(event)
            if day is None or not (start <= day < end):
                continue
            if event.category != category or event.direction is not EventDirection.DEBIT:
                continue
            if event.amount_home_currency is None:
                continue
            total += event.amount_home_currency
        cycles.append(total)
    return cycles


def production_config(repository: DatasetRepository) -> ForecastConfig:
    return forecast_config_from_profiles(
        repository.dataset.profiles,
        essential_spend_strategy=EssentialSpendStrategy.DAILY_RATE,
        essential_lookback_days=90,
        strict_unresolved_amounts=True,
    )
