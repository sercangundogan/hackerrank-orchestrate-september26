"""Experimental variable-spend reference models for analysis only.

Production forecast defaults are not imported from here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Callable

from data.models import EventDirection, EventType
from finance.essential_spending import (
    _explicit_projected,
    _historical_debits,
    _inliers,
    _select_lookback,
    eligible_essential_categories,
)
from finance.forecast import add_calendar_months
from finance.forecast_models import (
    ForecastCashEvent,
    ForecastConfig,
    ForecastEventKind,
    ForecastEventSource,
    RecurrenceProvenance,
    SameDayPriority,
)
from finance.models import Cadence, CadenceConfidence, NormalizedCashEvent, NormalizedFinancialState

_ZERO = Decimal("0")
_CONTRACTUAL = frozenset(
    {
        "rent",
        "debt_repayment",
        "insurance",
        "education",
        "housing",
        "family_support",
    }
)
_VARIABLE = frozenset({"groceries", "transport", "utilities", "healthcare"})


def _day(event: NormalizedCashEvent) -> date | None:
    return event.cash_date or event.event_date


def _completed_monthly_totals(
    events: tuple[NormalizedCashEvent, ...],
    request_date: date,
) -> tuple[Decimal, ...]:
    buckets: dict[tuple[int, int], Decimal] = defaultdict(lambda: _ZERO)
    if not events:
        return ()
    first = _day(events[0]) or request_date
    cursor = date(first.year, first.month, 1)
    last_complete = date(request_date.year, request_date.month, 1)
    # Exclude the in-progress request month.
    if last_complete > cursor:
        end = last_complete
    else:
        end = cursor
    while cursor < end:
        buckets[(cursor.year, cursor.month)] = _ZERO
        cursor = add_calendar_months(cursor, 1, pattern_day=1)
    for event in events:
        day = _day(event)
        if day is None or event.amount_home_currency is None:
            continue
        key = (day.year, day.month)
        if key in buckets:
            buckets[key] += event.amount_home_currency
    return tuple(buckets[key] for key in sorted(buckets))


def _typical_dom(events: tuple[NormalizedCashEvent, ...]) -> int:
    days = [(_day(event) or date.min).day for event in events if _day(event) is not None]
    if not days:
        return 15
    return int(median(days))


def _salary_dates(state: NormalizedFinancialState) -> list[date]:
    dates: list[date] = []
    for event in state.historical_settled_cash_events:
        day = _day(event)
        if day is None or day >= state.request_date:
            continue
        if event.category != "salary" or event.direction is not EventDirection.CREDIT:
            continue
        dates.append(day)
    return sorted(set(dates))


def _paycycle_totals(
    events: tuple[NormalizedCashEvent, ...],
    salaries: list[date],
) -> tuple[Decimal, ...]:
    if len(salaries) < 2:
        return ()
    totals: list[Decimal] = []
    for start, end in zip(salaries, salaries[1:]):
        total = _ZERO
        for event in events:
            day = _day(event)
            if day is None or event.amount_home_currency is None:
                continue
            if start <= day < end:
                total += event.amount_home_currency
        totals.append(total)
    return tuple(totals)


def _future_salary_dates(
    state: NormalizedFinancialState,
    existing: tuple[ForecastCashEvent, ...],
    horizon_end: date,
) -> list[date]:
    dates = [
        entry.date
        for entry in existing
        if entry.signed_amount > 0 and entry.category == "salary"
    ]
    historical = _salary_dates(state)
    if historical and not dates:
        last = historical[-1]
        gaps = [(b - a).days for a, b in zip(historical, historical[1:]) if (b - a).days > 0]
        step = int(median(gaps)) if gaps else 30
        cursor = last + timedelta(days=step)
        while cursor <= horizon_end:
            dates.append(cursor)
            cursor += timedelta(days=step)
    return sorted({item for item in dates if state.request_date <= item <= horizon_end})


def _make_entry(
    category: str,
    when: date,
    amount: Decimal,
    method: str,
) -> ForecastCashEvent:
    generated_id = f"essential:{category}:{when.isoformat()}:{method}"
    return ForecastCashEvent(
        date=when,
        amount_home_currency=amount,
        direction=EventDirection.DEBIT,
        signed_amount=-amount,
        kind=ForecastEventKind.ESSENTIAL_SPEND_RESERVE,
        source=ForecastEventSource.CATEGORY_RESERVE,
        source_event_id=generated_id,
        source_series_key=f"essential:{category}",
        is_generated_recurrence=True,
        priority=SameDayPriority.GENERATED_RECURRING_DEBIT,
        description=f"essential residual {category} {method}",
        provenance=RecurrenceProvenance(
            series_key=f"essential:{category}",
            source_event_ids=(),
            generation_method=method,
            cadence=Cadence.MONTHLY,
            amount_strategy=method,
            confidence=CadenceConfidence.MEDIUM,
        ),
        category=category,
        order_key=generated_id,
    )


def _allocate(dates: list[date], residual: Decimal) -> list[tuple[date, Decimal]]:
    if not dates or residual <= _ZERO:
        return []
    exponent = residual.as_tuple().exponent
    quantum = Decimal("1") if isinstance(exponent, int) and exponent >= 0 else Decimal("0.01")
    share = (residual / Decimal(len(dates))).quantize(quantum)
    allocated = _ZERO
    out: list[tuple[date, Decimal]] = []
    for index, when in enumerate(dates):
        amount = residual - allocated if index == len(dates) - 1 else share
        if amount > _ZERO:
            allocated += amount
            out.append((when, amount))
    return out


def _eligible_split_categories(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing: tuple[ForecastCashEvent, ...],
) -> list[tuple[str, tuple[NormalizedCashEvent, ...], Decimal]]:
    eligible = eligible_essential_categories(state.profile, config.extra_essential_categories)
    rows: list[tuple[str, tuple[NormalizedCashEvent, ...], Decimal]] = []
    for category in sorted(eligible):
        history = _historical_debits(state, category, state.request_date)
        window, _start = _select_lookback(history, state.request_date, config.essential_lookback_days)
        inliers, _outliers = _inliers(window)
        descriptions = {event.description for event in inliers}
        if len(descriptions) < 2:
            continue
        explicit = _explicit_projected(existing, category)
        rows.append((category, inliers, explicit))
    return rows


def _monthly_dates(state: NormalizedFinancialState, horizon_end: date, typical_dom: int) -> list[date]:
    dates: list[date] = []
    cursor = date(state.request_date.year, state.request_date.month, 1)
    while cursor <= horizon_end:
        lump = add_calendar_months(cursor, 0, pattern_day=typical_dom)
        if state.request_date <= lump <= horizon_end:
            dates.append(lump)
        elif lump < state.request_date <= horizon_end and cursor.month == state.request_date.month:
            dates.append(state.request_date)
        cursor = add_calendar_months(cursor, 1, pattern_day=1)
    return dates


def builder_monthly(
    how: str,
) -> Callable[[NormalizedFinancialState, ForecastConfig, tuple[ForecastCashEvent, ...]], list[ForecastCashEvent]]:
    def build(
        state: NormalizedFinancialState,
        config: ForecastConfig,
        existing: tuple[ForecastCashEvent, ...],
    ) -> list[ForecastCashEvent]:
        horizon_end = config.horizon_end(state.request_date)
        entries: list[ForecastCashEvent] = []
        for category, inliers, explicit in _eligible_split_categories(state, config, existing):
            months = _completed_monthly_totals(inliers, state.request_date)
            if not months:
                continue
            if how == "A":
                budget = sum(months, _ZERO) / Decimal(len(months))
            elif how == "B":
                recent = months[-3:]
                budget = sum(recent, _ZERO) / Decimal(len(recent))
            else:
                budget = max(months[-3:])
            dates = _monthly_dates(state, horizon_end, _typical_dom(inliers))
            residual = max(_ZERO, budget * Decimal(len(dates)) - explicit)
            for when, amount in _allocate(dates, residual):
                entries.append(_make_entry(category, when, amount, f"monthly_{how}"))
        return entries

    return build


def builder_cadence(
    amount_mode: str,
) -> Callable[[NormalizedFinancialState, ForecastConfig, tuple[ForecastCashEvent, ...]], list[ForecastCashEvent]]:
    def build(
        state: NormalizedFinancialState,
        config: ForecastConfig,
        existing: tuple[ForecastCashEvent, ...],
    ) -> list[ForecastCashEvent]:
        horizon_end = config.horizon_end(state.request_date)
        entries: list[ForecastCashEvent] = []
        for category, inliers, explicit in _eligible_split_categories(state, config, existing):
            days = [_day(event) for event in inliers if _day(event) is not None]
            if len(days) < 2:
                continue
            gaps = [(b - a).days for a, b in zip(days, days[1:]) if (b - a).days > 0]
            if not gaps:
                continue
            gap = max(1, int(median(gaps)))
            amounts = [event.amount_home_currency or _ZERO for event in inliers]
            recent = amounts[-5:]
            if amount_mode == "mean":
                unit = sum(recent, _ZERO) / Decimal(len(recent))
            elif amount_mode == "max":
                unit = max(recent)
            else:
                unit = Decimal(str(median(recent)))
            cursor = days[-1] + timedelta(days=gap)
            if cursor < state.request_date:
                skipped = (state.request_date - cursor).days // gap + 1
                cursor = cursor + timedelta(days=skipped * gap)
            dates: list[date] = []
            while cursor <= horizon_end:
                dates.append(cursor)
                cursor += timedelta(days=gap)
            residual = max(_ZERO, unit * Decimal(len(dates)) - explicit)
            for when, amount in _allocate(dates, residual):
                entries.append(_make_entry(category, when, amount, f"cadence_{amount_mode}"))
        return entries

    return build


def builder_paycycle(
    how: str,
) -> Callable[[NormalizedFinancialState, ForecastConfig, tuple[ForecastCashEvent, ...]], list[ForecastCashEvent]]:
    def build(
        state: NormalizedFinancialState,
        config: ForecastConfig,
        existing: tuple[ForecastCashEvent, ...],
    ) -> list[ForecastCashEvent]:
        horizon_end = config.horizon_end(state.request_date)
        salaries = _salary_dates(state)
        future = _future_salary_dates(state, existing, horizon_end)
        entries: list[ForecastCashEvent] = []
        for category, inliers, explicit in _eligible_split_categories(state, config, existing):
            cycles = _paycycle_totals(inliers, salaries)
            if cycles:
                recent = cycles[-3:]
                if how == "latest":
                    budget = cycles[-1]
                elif how == "avg":
                    budget = sum(recent, _ZERO) / Decimal(len(recent))
                else:
                    budget = max(recent)
            else:
                months = _completed_monthly_totals(inliers, state.request_date)
                if not months:
                    continue
                budget = sum(months[-3:], _ZERO) / Decimal(max(1, len(months[-3:])))
            dates: list[date] = [state.request_date]
            for payday in future:
                spend_day = payday + timedelta(days=1)
                if state.request_date < spend_day <= horizon_end:
                    dates.append(spend_day)
            dates = sorted(set(dates))
            residual = max(_ZERO, budget * Decimal(len(dates)) - explicit)
            for when, amount in _allocate(dates, residual):
                entries.append(_make_entry(category, when, amount, f"paycycle_{how}"))
        return entries

    return build


def builder_hist90_weekly(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing: tuple[ForecastCashEvent, ...],
) -> list[ForecastCashEvent]:
    horizon_end = config.horizon_end(state.request_date)
    horizon_days = (horizon_end - state.request_date).days
    entries: list[ForecastCashEvent] = []
    for category, inliers, explicit in _eligible_split_categories(state, config, existing):
        total = sum((event.amount_home_currency or _ZERO for event in inliers), _ZERO)
        expected = total * (Decimal(horizon_days) / Decimal(max(1, config.essential_lookback_days)))
        residual = max(_ZERO, expected - explicit)
        dates: list[date] = []
        cursor = state.request_date
        while cursor <= horizon_end:
            dates.append(cursor)
            cursor += timedelta(days=7)
        for when, amount in _allocate(dates, residual):
            entries.append(_make_entry(category, when, amount, "hist90_weekly"))
    return entries


def _first_income_date(
    existing: tuple[ForecastCashEvent, ...],
    request_date: date,
) -> date | None:
    dates = [entry.date for entry in existing if entry.signed_amount > 0 and entry.date >= request_date]
    return min(dates) if dates else None


def builder_frontload_prepayday(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing: tuple[ForecastCashEvent, ...],
) -> list[ForecastCashEvent]:
    """Put the pre-income share of the 90-day residual on request_date."""
    horizon_end = config.horizon_end(state.request_date)
    horizon_days = max(1, (horizon_end - state.request_date).days)
    first_income = _first_income_date(existing, state.request_date)
    window = (first_income - state.request_date).days if first_income else min(30, horizon_days)
    window = max(1, window)
    entries: list[ForecastCashEvent] = []
    for category, inliers, explicit in _eligible_split_categories(state, config, existing):
        total = sum((event.amount_home_currency or _ZERO for event in inliers), _ZERO)
        expected = total * (Decimal(horizon_days) / Decimal(max(1, config.essential_lookback_days)))
        residual = max(_ZERO, expected - explicit)
        share = residual * Decimal(window) / Decimal(horizon_days)
        later = residual - share
        if share > _ZERO:
            entries.append(_make_entry(category, state.request_date, share, "frontload_prepayday"))
        dates: list[date] = []
        cursor = state.request_date + timedelta(days=7)
        while cursor <= horizon_end:
            dates.append(cursor)
            cursor += timedelta(days=7)
        for when, amount in _allocate(dates, later):
            entries.append(_make_entry(category, when, amount, "frontload_tail"))
    return entries


def builder_relative_payday(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing: tuple[ForecastCashEvent, ...],
) -> list[ForecastCashEvent]:
    """Place category residuals on historical relative-to-payday days."""
    horizon_end = config.horizon_end(state.request_date)
    first_income = _first_income_date(existing, state.request_date)
    salaries = [
        entry.date
        for entry in existing
        if entry.signed_amount > 0 and entry.category == "salary"
    ]
    if first_income is None:
        return builder_hist90_weekly(state, config, existing)
    entries: list[ForecastCashEvent] = []
    for category, inliers, explicit in _eligible_split_categories(state, config, existing):
        offsets: list[int] = []
        hist_salaries = _salary_dates(state)
        for event in inliers:
            day = _day(event)
            if day is None:
                continue
            nearby = [pay for pay in hist_salaries if abs((day - pay).days) <= 20]
            if not nearby:
                continue
            pay = min(nearby, key=lambda item: abs((day - item).days))
            offsets.append((day - pay).days)
        if not offsets:
            return_dates = [state.request_date + timedelta(days=7 * i) for i in range(13)]
            return_dates = [item for item in return_dates if item <= horizon_end]
        else:
            typical = int(median(offsets))
            return_dates = []
            for payday in salaries or [first_income]:
                when = payday + timedelta(days=typical)
                if state.request_date <= when <= horizon_end:
                    return_dates.append(when)
            if not return_dates:
                when = first_income + timedelta(days=typical)
                if state.request_date <= when <= horizon_end:
                    return_dates.append(when)
                else:
                    return_dates.append(state.request_date)
        total = sum((event.amount_home_currency or _ZERO for event in inliers), _ZERO)
        horizon_days = max(1, (horizon_end - state.request_date).days)
        expected = total * (Decimal(horizon_days) / Decimal(max(1, config.essential_lookback_days)))
        residual = max(_ZERO, expected - explicit)
        for when, amount in _allocate(sorted(set(return_dates)), residual):
            entries.append(_make_entry(category, when, amount, "relative_payday"))
    return entries


def builder_none(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing: tuple[ForecastCashEvent, ...],
) -> list[ForecastCashEvent]:
    del state, config, existing
    return []


EXPERIMENTAL_BUILDERS: dict[
    str,
    Callable[[NormalizedFinancialState, ForecastConfig, tuple[ForecastCashEvent, ...]], list[ForecastCashEvent]],
] = {
    "none": builder_none,
    "G_hist90_weekly": builder_hist90_weekly,
    "A_monthly_mean_all": builder_monthly("A"),
    "B_monthly_mean_last3": builder_monthly("B"),
    "C_monthly_max_last3": builder_monthly("C"),
    "D_cadence_median": builder_cadence("median"),
    "D_cadence_mean": builder_cadence("mean"),
    "D_cadence_max": builder_cadence("max"),
    "E_paycycle_avg": builder_paycycle("avg"),
    "E_paycycle_latest": builder_paycycle("latest"),
    "F_paycycle_max": builder_paycycle("max"),
    "prepayday_frontload": builder_frontload_prepayday,
    "relative_payday": builder_relative_payday,
}


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    amount_exact: int
    earliest_exact: int
    mae: Decimal
    median_norm: Decimal
    within_1: int
    within_5: int
    within_10: int
    over: int
    under: int
    floor_hits: int
    n: int = 25


def summarize_capacity_rows(name: str, rows, labels_near_floor: int) -> ModelMetrics:
    n = len(rows)
    errors = [abs(row.amount_error) for row in rows]
    norms = [
        (abs(row.amount_error) / row.requested_amount) if row.requested_amount else Decimal("0")
        for row in rows
    ]
    norms_sorted = sorted(norms)
    mid = norms_sorted[n // 2] if n else Decimal("0")
    return ModelMetrics(
        name=name,
        amount_exact=sum(1 for row in rows if row.amount_match),
        earliest_exact=sum(1 for row in rows if row.date_match),
        mae=sum(errors, _ZERO) / Decimal(n),
        median_norm=mid,
        within_1=sum(1 for item in norms if item <= Decimal("0.01")),
        within_5=sum(1 for item in norms if item <= Decimal("0.05")),
        within_10=sum(1 for item in norms if item <= Decimal("0.10")),
        over=sum(1 for row in rows if row.amount_error > 0),
        under=sum(1 for row in rows if row.amount_error < 0),
        floor_hits=labels_near_floor,
        n=n,
    )
