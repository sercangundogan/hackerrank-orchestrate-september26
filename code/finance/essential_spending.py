"""Category-level residual essential spending.

Complements description-level recurrence. Protected and commonly protected
expense categories can still reserve cash when history is split across
descriptions and therefore never forms a HIGH/MEDIUM series.

Residual = max(0, expected_category_spend − already_projected_category_spend).
Income never enters this reserve. One-off outliers are dropped.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from data.models import EventDirection, EventType, FinancialProfile, Flexibility
from finance.forecast_models import (
    EssentialSpendStrategy,
    ForecastCashEvent,
    ForecastConfig,
    ForecastEventKind,
    ForecastEventSource,
    RecurrenceProvenance,
    SameDayPriority,
)
from finance.models import Cadence, CadenceConfidence, NormalizedCashEvent, NormalizedFinancialState

_ZERO = Decimal("0")
_OUTLIER_MULT = Decimal("3")
_MIN_INLIER_EVENTS = 3


def infer_common_protected_categories(profiles: tuple[FinancialProfile, ...]) -> frozenset[str]:
    """Categories many profiles protect — inferred from data, not English lists."""
    counts: Counter[str] = Counter()
    for profile in profiles:
        for category in profile.expense_categories_to_protect:
            counts[category] += 1
    if not profiles:
        return frozenset()
    floor = max(8, len(profiles) // 20)
    return frozenset(category for category, count in counts.items() if count >= floor)


def forecast_config_from_profiles(
    profiles: tuple[FinancialProfile, ...],
    **overrides: object,
) -> ForecastConfig:
    extras = infer_common_protected_categories(profiles)
    values: dict[str, object] = {"extra_essential_categories": extras}
    values.update(overrides)
    return ForecastConfig(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CategorySpendingProfile:
    category: str
    historical_event_ids: tuple[str, ...]
    lookback_start: date
    lookback_end: date
    total_spend: Decimal
    weekly_totals: tuple[Decimal, ...]
    monthly_totals: tuple[Decimal, ...]
    projected_budget: Decimal
    explicit_projected_amount: Decimal
    residual_reserve: Decimal
    strategy: EssentialSpendStrategy
    confidence: CadenceConfidence
    excluded_outlier_ids: tuple[str, ...]


def _event_day(event: NormalizedCashEvent) -> date | None:
    return event.cash_date or event.event_date


def _is_investment(event: NormalizedCashEvent) -> bool:
    return event.event_type in {
        EventType.INVESTMENT_PURCHASE,
        EventType.INVESTMENT_SALE,
        EventType.INVESTMENT_VALUATION,
    }


def eligible_essential_categories(
    profile: FinancialProfile,
    extra: frozenset[str],
) -> frozenset[str]:
    eligible = set(profile.expense_categories_to_protect)
    eligible |= set(extra)
    eligible -= set(profile.expense_categories_user_is_willing_to_stop)
    eligible -= set(profile.expense_categories_user_is_willing_to_reduce)
    eligible |= set(profile.expense_categories_to_protect)
    return frozenset(eligible)


def _historical_debits(
    state: NormalizedFinancialState,
    category: str,
    request_date: date,
) -> tuple[NormalizedCashEvent, ...]:
    rows: list[NormalizedCashEvent] = []
    for event in state.historical_settled_cash_events:
        day = _event_day(event)
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
        if _is_investment(event):
            continue
        if event.flexibility in {Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE}:
            continue
        rows.append(event)
    rows.sort(key=lambda item: (_event_day(item) or request_date, item.source_event_id))
    return tuple(rows)


def _inliers(
    events: tuple[NormalizedCashEvent, ...],
) -> tuple[tuple[NormalizedCashEvent, ...], tuple[str, ...]]:
    amounts = tuple(
        event.amount_home_currency
        for event in events
        if event.amount_home_currency is not None
    )
    if len(amounts) < 4:
        if len(amounts) < _MIN_INLIER_EVENTS:
            return (), tuple(event.source_event_id for event in events)
        return events, ()
    mid = Decimal(str(median(amounts)))
    cap = mid * _OUTLIER_MULT
    kept: list[NormalizedCashEvent] = []
    dropped: list[str] = []
    for event in events:
        amount = event.amount_home_currency or _ZERO
        if amount > cap:
            dropped.append(event.source_event_id)
        else:
            kept.append(event)
    if len(kept) < _MIN_INLIER_EVENTS:
        return (), tuple(event.source_event_id for event in events)
    return tuple(kept), tuple(dropped)


def _select_lookback(
    events: tuple[NormalizedCashEvent, ...],
    request_date: date,
    lookback_days: int,
) -> tuple[tuple[NormalizedCashEvent, ...], date]:
    start = request_date - timedelta(days=lookback_days)
    window = tuple(
        event
        for event in events
        if start <= (_event_day(event) or request_date) < request_date
    )
    if len(window) >= _MIN_INLIER_EVENTS:
        return window, start
    tail = events[-12:]
    if not tail:
        return (), start
    first = _event_day(tail[0]) or start
    return tail, first


def _weekly_totals(
    events: tuple[NormalizedCashEvent, ...],
    start: date,
    end: date,
) -> tuple[Decimal, ...]:
    buckets: dict[date, Decimal] = defaultdict(lambda: _ZERO)
    cursor = start
    while cursor < end:
        buckets[cursor] = _ZERO
        cursor += timedelta(days=7)
    for event in events:
        day = _event_day(event)
        if day is None or event.amount_home_currency is None:
            continue
        offset = max(0, (day - start).days)
        week_start = start + timedelta(days=(offset // 7) * 7)
        buckets[week_start] += event.amount_home_currency
    return tuple(buckets[key] for key in sorted(buckets))


def _monthly_totals(
    events: tuple[NormalizedCashEvent, ...],
    start: date,
    end: date,
) -> tuple[Decimal, ...]:
    buckets: dict[tuple[int, int], Decimal] = defaultdict(lambda: _ZERO)
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        buckets[(cursor.year, cursor.month)] = _ZERO
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    for event in events:
        day = _event_day(event)
        if day is None or event.amount_home_currency is None:
            continue
        buckets[(day.year, day.month)] += event.amount_home_currency
    return tuple(buckets[key] for key in sorted(buckets))


def _recent_stat(values: tuple[Decimal, ...], *, use_max: bool, window: int = 4) -> Decimal:
    nonzero = tuple(item for item in values if item > _ZERO)
    recent = nonzero[-window:] if nonzero else (values[-window:] if values else ())
    if not recent:
        return _ZERO
    if use_max:
        return max(recent)
    return Decimal(str(median(recent)))


def expected_category_spend(
    *,
    total: Decimal,
    lookback_days: int,
    horizon_days: int,
    weekly: tuple[Decimal, ...],
    monthly: tuple[Decimal, ...],
    strategy: EssentialSpendStrategy,
    event_count: int,
    span_days: int,
) -> Decimal:
    if lookback_days <= 0 or horizon_days <= 0 or event_count < _MIN_INLIER_EVENTS:
        return _ZERO
    horizon = Decimal(horizon_days)
    lookback = Decimal(lookback_days)
    if strategy is EssentialSpendStrategy.DAILY_RATE:
        return (total / lookback) * horizon
    if strategy is EssentialSpendStrategy.WEEKLY_MEDIAN:
        return _recent_stat(weekly, use_max=False) * (horizon / Decimal(7))
    if strategy is EssentialSpendStrategy.WEEKLY_MAX:
        return _recent_stat(weekly, use_max=True) * (horizon / Decimal(7))
    if strategy is EssentialSpendStrategy.MONTHLY_MAX:
        return _recent_stat(monthly, use_max=True, window=3) * (horizon / Decimal(30))
    # HYBRID: weekly-ish history uses weekly median; otherwise monthly max.
    typical_gap = span_days / max(event_count - 1, 1)
    if typical_gap <= 12:
        return _recent_stat(weekly, use_max=False) * (horizon / Decimal(7))
    return _recent_stat(monthly, use_max=True, window=3) * (horizon / Decimal(30))


def _explicit_projected(
    entries: tuple[ForecastCashEvent, ...],
    category: str,
) -> Decimal:
    total = _ZERO
    for entry in entries:
        if entry.category != category:
            continue
        if entry.signed_amount >= 0:
            continue
        if entry.kind is ForecastEventKind.CANDIDATE_PAYMENT:
            continue
        if entry.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE:
            continue
        total += entry.amount_home_currency
    return total


def build_category_profiles(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing_entries: tuple[ForecastCashEvent, ...],
) -> tuple[CategorySpendingProfile, ...]:
    if not config.essential_spend_enabled:
        return ()
    eligible = eligible_essential_categories(
        state.profile, config.extra_essential_categories
    )
    horizon_days = (config.horizon_end(state.request_date) - state.request_date).days
    profiles: list[CategorySpendingProfile] = []
    for category in sorted(eligible):
        history = _historical_debits(state, category, state.request_date)
        window, start = _select_lookback(history, state.request_date, config.essential_lookback_days)
        inliers, outliers = _inliers(window)
        total = sum((event.amount_home_currency or _ZERO for event in inliers), _ZERO)
        weekly = _weekly_totals(inliers, start, state.request_date)
        monthly = _monthly_totals(inliers, start, state.request_date)
        first = _event_day(inliers[0]) if inliers else start
        last = _event_day(inliers[-1]) if inliers else start
        span = max(1, (last - first).days) if inliers else 1
        expected = expected_category_spend(
            total=total,
            lookback_days=max(1, (state.request_date - start).days),
            horizon_days=horizon_days,
            weekly=weekly,
            monthly=monthly,
            strategy=config.essential_spend_strategy,
            event_count=len(inliers),
            span_days=span,
        )
        explicit = _explicit_projected(existing_entries, category)
        descriptions = {event.description for event in inliers}
        residual = expected - explicit
        if residual < _ZERO:
            residual = _ZERO
        # Description-level recurrence already owns single-description series.
        # Category reserves exist to recombine split descriptions.
        if len(descriptions) < 2:
            residual = _ZERO
        confidence = (
            CadenceConfidence.HIGH
            if len(inliers) >= 8
            else CadenceConfidence.MEDIUM
            if len(inliers) >= _MIN_INLIER_EVENTS
            else CadenceConfidence.LOW
        )
        profiles.append(
            CategorySpendingProfile(
                category=category,
                historical_event_ids=tuple(event.source_event_id for event in inliers),
                lookback_start=start,
                lookback_end=state.request_date,
                total_spend=total,
                weekly_totals=weekly,
                monthly_totals=monthly,
                projected_budget=expected,
                explicit_projected_amount=explicit,
                residual_reserve=residual,
                strategy=config.essential_spend_strategy,
                confidence=confidence,
                excluded_outlier_ids=outliers,
            )
        )
    return tuple(profiles)


def essential_spend_entries(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    existing_entries: tuple[ForecastCashEvent, ...],
) -> list[ForecastCashEvent]:
    profiles = build_category_profiles(state, config, existing_entries)
    horizon_end = config.horizon_end(state.request_date)
    entries: list[ForecastCashEvent] = []
    for profile in profiles:
        if profile.residual_reserve <= _ZERO:
            continue
        dates: list[date] = []
        cursor = state.request_date
        while cursor <= horizon_end:
            dates.append(cursor)
            cursor += timedelta(days=7)
        if not dates:
            continue
        exponent = profile.residual_reserve.as_tuple().exponent
        quantum = Decimal("1") if isinstance(exponent, int) and exponent >= 0 else Decimal("0.01")
        share = (profile.residual_reserve / Decimal(len(dates))).quantize(quantum)
        allocated = _ZERO
        for index, when in enumerate(dates):
            amount = profile.residual_reserve - allocated if index == len(dates) - 1 else share
            if amount <= _ZERO:
                continue
            allocated += amount
            generated_id = f"essential:{profile.category}:{when.isoformat()}"
            entries.append(
                ForecastCashEvent(
                    date=when,
                    amount_home_currency=amount,
                    direction=EventDirection.DEBIT,
                    signed_amount=-amount,
                    kind=ForecastEventKind.ESSENTIAL_SPEND_RESERVE,
                    source=ForecastEventSource.CATEGORY_RESERVE,
                    source_event_id=generated_id,
                    source_series_key=f"essential:{profile.category}",
                    is_generated_recurrence=True,
                    priority=SameDayPriority.GENERATED_RECURRING_DEBIT,
                    description=(
                        f"essential residual {profile.category} "
                        f"{profile.strategy.value} {profile.confidence.value}"
                    ),
                    provenance=RecurrenceProvenance(
                        series_key=f"essential:{profile.category}",
                        source_event_ids=profile.historical_event_ids,
                        generation_method="category_residual",
                        cadence=Cadence.WEEKLY,
                        amount_strategy=profile.strategy.value,
                        confidence=profile.confidence,
                    ),
                    category=profile.category,
                    order_key=generated_id,
                )
            )
    return entries
