"""Deterministic 90-day cash-flow simulator.

Starts from the profile snapshot. Does not replay historical settled cash,
compute a safe payment amount, classify affordability, or recommend a method.
"""

from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from data.models import EventDirection, EventStatus, EventType
from finance.forecast_models import (
    ForecastCashEvent,
    ForecastConfig,
    ForecastEventKind,
    ForecastEventSource,
    ForecastResult,
    DailyForecast,
    PendingDebitPolicy,
    RecurrenceOverride,
    RecurrenceProvenance,
    SalaryProjectionMode,
    SameDayPriority,
    UnresolvedForecastError,
    VariableAmountStrategy,
)
from finance.adjustments import ForecastAdjustment, ForecastAdjustmentKind
from finance.models import (
    Cadence,
    CadenceConfidence,
    NormalizedCashEvent,
    NormalizedFinancialState,
    RecurringSeriesCandidate,
)
from finance.essential_spending import essential_spend_entries
from finance.income import (
    amendment_targets_series,
    classify_income_text,
    classify_series,
    continue_start_salary,
    is_base_payroll,
    never_project_subtype,
    salary_generation_allowed,
)
from finance.recurrence import normalize_description

_ZERO = Decimal("0")
_RECENT_MAX_DEFAULT_WINDOW = 3


def add_calendar_months(anchor: date, months: int, *, pattern_day: int | None = None) -> date:
    """Advance `months` calendar months without 30-day drift.

    `pattern_day` is the intended day-of-month (defaults to `anchor.day`).
    Month-end anchors clamp to the last valid day of the target month, then
    restore the original pattern day on later longer months (Jan 31 → Feb 28
    → Mar 31).
    """
    if months == 0:
        day = pattern_day if pattern_day is not None else anchor.day
        last = calendar.monthrange(anchor.year, anchor.month)[1]
        return date(anchor.year, anchor.month, min(day, last))
    month_index = anchor.month - 1 + months
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = pattern_day if pattern_day is not None else anchor.day
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, last))


def next_recurrence_date(
    previous: date,
    cadence: Cadence,
    *,
    pattern_day: int | None = None,
    cadence_days: int | None = None,
) -> date:
    """Return the next occurrence after `previous` for a known cadence."""
    if cadence is Cadence.WEEKLY:
        return previous + timedelta(days=7)
    if cadence is Cadence.BIWEEKLY:
        return previous + timedelta(days=14)
    if cadence is Cadence.MONTHLY:
        return add_calendar_months(previous, 1, pattern_day=pattern_day or previous.day)
    raise ValueError(f"cannot project cadence {cadence.value}")


def iter_recurrence_dates(
    *,
    last_observed: date,
    cadence: Cadence,
    horizon_start: date,
    horizon_end: date,
    pattern_day: int | None = None,
    cadence_days: int | None = None,
) -> tuple[date, ...]:
    """Generate dates after `last_observed` inside the inclusive horizon."""
    if cadence not in {Cadence.WEEKLY, Cadence.BIWEEKLY, Cadence.MONTHLY}:
        return ()
    day = pattern_day if pattern_day is not None else last_observed.day
    cursor = last_observed
    found: list[date] = []
    # Hard cap: weekly over 91 days is < 20; monthly < 5. Use a generous bound.
    for _ in range(64):
        cursor = next_recurrence_date(
            cursor,
            cadence,
            pattern_day=day,
            cadence_days=cadence_days,
        )
        if cursor > horizon_end:
            break
        if cursor >= horizon_start:
            found.append(cursor)
    return tuple(found)


def resolve_variable_amount(
    amounts: tuple[Decimal | None, ...],
    strategy: VariableAmountStrategy,
    *,
    window: int = _RECENT_MAX_DEFAULT_WINDOW,
) -> Decimal | None:
    """Resolve a projected amount. Never substitutes 0 for a missing series."""
    known = tuple(amount for amount in amounts if amount is not None)
    if not known:
        return None
    if strategy is VariableAmountStrategy.LATEST:
        return known[-1]
    if strategy is VariableAmountStrategy.MAX:
        return max(known)
    if strategy is VariableAmountStrategy.RECENT_MAX:
        recent = known[-max(1, window) :]
        return max(recent)
    raise ValueError(f"unknown variable amount strategy {strategy}")


def series_key(series: RecurringSeriesCandidate) -> str:
    return (
        f"{series.user_id}|{series.category}|{series.normalized_description}|"
        f"{series.direction.value}"
    )


def event_series_key(event: NormalizedCashEvent) -> str:
    return (
        f"{event.user_id}|{event.category}|{normalize_description(event.description)}|"
        f"{event.direction.value}"
    )


def _signed(direction: EventDirection, amount: Decimal) -> Decimal:
    if direction is EventDirection.CREDIT:
        return amount
    if direction is EventDirection.DEBIT:
        return -amount
    return _ZERO


def _in_horizon(when: date, start: date, end: date) -> bool:
    return start <= when <= end


def _proximity_days(cadence: Cadence) -> int:
    if cadence is Cadence.WEEKLY:
        return 2
    if cadence is Cadence.BIWEEKLY:
        return 3
    if cadence is Cadence.MONTHLY:
        return 4
    return 2


def _is_salary_series(series: RecurringSeriesCandidate) -> bool:
    return series.category == "salary" and series.direction is EventDirection.CREDIT


def _is_salary_event(event: NormalizedCashEvent) -> bool:
    return event.category == "salary" and event.direction is EventDirection.CREDIT


def _eligible_for_auto_projection(
    series: RecurringSeriesCandidate,
    config: ForecastConfig,
) -> bool:
    if series.inferred_cadence not in {Cadence.WEEKLY, Cadence.BIWEEKLY, Cadence.MONTHLY}:
        return False
    if series.cadence_confidence is CadenceConfidence.HIGH:
        return True
    if (
        series.cadence_confidence is CadenceConfidence.MEDIUM
        and config.include_medium_confidence_recurrence
    ):
        return True
    # LOW irregular/flexible series stay out of the baseline forecast.
    # Subscriptions already marked likely recurring may still project when
    # they have a usable calendar cadence — flexibility alone is not enough.
    if (
        series.event_type is EventType.SUBSCRIPTION
        and series.likely_recurring
        and series.inferred_cadence in {Cadence.WEEKLY, Cadence.BIWEEKLY, Cadence.MONTHLY}
    ):
        return True
    return False


def _salary_history_eligible(
    series: RecurringSeriesCandidate,
    config: ForecastConfig,
) -> bool:
    if not _is_salary_series(series):
        return False
    if config.salary_projection_mode is not SalaryProjectionMode.SCHEDULED_PLUS_REGULAR_HISTORY:
        return False
    if series.inferred_cadence is not Cadence.MONTHLY:
        return _eligible_for_auto_projection(series, config)
    if series.cadence_confidence is CadenceConfidence.HIGH:
        return True
    return (
        series.cadence_confidence is CadenceConfidence.MEDIUM
        and config.include_medium_confidence_recurrence
    )


def _projected_amount(
    series: RecurringSeriesCandidate,
    config: ForecastConfig,
) -> Decimal | None:
    if series.amount_behavior.value == "fixed" and series.representative_amount is not None:
        return series.representative_amount
    # The configured strategy is for variable expenses. Inflating future
    # income with MAX / RECENT_MAX would silently assume a raise or rebound.
    strategy = config.variable_amount_strategy
    if series.direction is EventDirection.CREDIT:
        strategy = VariableAmountStrategy.LATEST
    return resolve_variable_amount(
        series.observed_amounts_home_currency,
        strategy,
        window=config.recent_max_window,
    )


def _sort_key(entry: ForecastCashEvent) -> tuple[int, str, str, str]:
    return (
        int(entry.priority),
        entry.kind.value,
        entry.order_key or entry.source_event_id or entry.source_series_key or "",
        entry.description,
    )


def _explicit_occupancy(
    state: NormalizedFinancialState,
) -> tuple[NormalizedCashEvent, ...]:
    """Pending/scheduled rows that already represent a future occurrence."""
    occupied: list[NormalizedCashEvent] = []
    for event in state.classified_events:
        if event.status not in {EventStatus.PENDING, EventStatus.SCHEDULED}:
            continue
        if event.direction is EventDirection.NON_CASH:
            continue
        occupied.append(event)
    return tuple(occupied)


def _matches_series(event: NormalizedCashEvent, series: RecurringSeriesCandidate) -> bool:
    if event.direction is not series.direction:
        return False
    if _is_salary_series(series) and _is_salary_event(event):
        event_subtype = classify_income_text(event.description, event.category)
        series_subtype = classify_series(series)
        if never_project_subtype(event_subtype) or never_project_subtype(series_subtype):
            return normalize_description(event.description) == series.normalized_description
        if is_base_payroll(event_subtype) and is_base_payroll(series_subtype):
            generic = "confirmed salary" in (event.description or "").lower()
            return generic or normalize_description(event.description) == series.normalized_description
        return normalize_description(event.description) == series.normalized_description
    if event.category != series.category:
        return False
    return normalize_description(event.description) == series.normalized_description


def _is_duplicate_of_explicit(
    *,
    series: RecurringSeriesCandidate,
    generated_date: date,
    occupied: tuple[NormalizedCashEvent, ...],
) -> bool:
    radius = _proximity_days(series.inferred_cadence)
    for event in occupied:
        if not _matches_series(event, series):
            continue
        event_date = event.cash_date or event.settlement_date or event.event_date
        if abs((event_date - generated_date).days) <= radius:
            return True
    return False


def _pending_debit_entries(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    horizon_start: date,
    unresolved: list[str],
) -> list[ForecastCashEvent]:
    entries: list[ForecastCashEvent] = []
    if config.pending_debit_policy is not PendingDebitPolicy.IMMEDIATE_RESERVE:
        raise ValueError(f"unsupported pending debit policy {config.pending_debit_policy}")
    for event in state.pending_debits:
        if event.amount_home_currency is None:
            unresolved.append(
                f"pending debit {event.source_event_id} has an unresolved amount "
                "and cannot be reserved as zero"
            )
            continue
        entries.append(
            ForecastCashEvent(
                date=horizon_start,
                amount_home_currency=event.amount_home_currency,
                direction=EventDirection.DEBIT,
                signed_amount=_signed(EventDirection.DEBIT, event.amount_home_currency),
                kind=ForecastEventKind.PENDING_DEBIT_RESERVE,
                source=ForecastEventSource.RESERVED_PENDING,
                source_event_id=event.source_event_id,
                source_series_key=event_series_key(event),
                is_generated_recurrence=False,
                priority=SameDayPriority.OBLIGATED_DEBIT,
                description=(
                    f"reserve pending debit {event.source_event_id} "
                    f"({event.description}; original cash_date="
                    f"{event.cash_date.isoformat() if event.cash_date else 'none'}, "
                    f"event_date={event.event_date.isoformat()})"
                ),
                original_event_date=event.event_date,
                original_settlement_date=event.settlement_date,
                original_cash_date=event.cash_date,
                flexibility=event.flexibility,
                category=event.category,
                order_key=event.source_event_id,
            )
        )
    return entries


def override_for_series(
    series: RecurringSeriesCandidate,
    overrides: tuple[RecurrenceOverride, ...],
) -> RecurrenceOverride | None:
    """Match an override to a series by any event id in that series."""
    if not overrides:
        return None
    ids = set(series.event_ids)
    for item in overrides:
        if item.event_id in ids:
            return item
    return None


def override_for_event(
    event: NormalizedCashEvent,
    series_list: tuple[RecurringSeriesCandidate, ...],
    overrides: tuple[RecurrenceOverride, ...],
) -> RecurrenceOverride | None:
    if not overrides:
        return None
    for series in series_list:
        if event.source_event_id in series.event_ids:
            found = override_for_series(series, overrides)
            if found is not None:
                return found
        if event_series_key(event) == series_key(series):
            found = override_for_series(series, overrides)
            if found is not None:
                return found
    return None


def _scheduled_debit_entries(
    state: NormalizedFinancialState,
    horizon_start: date,
    horizon_end: date,
    unresolved: list[str],
    overrides: tuple[RecurrenceOverride, ...] = (),
) -> list[ForecastCashEvent]:
    entries: list[ForecastCashEvent] = []
    for event in state.scheduled_debits:
        cash_date = event.cash_date or event.settlement_date
        if cash_date is None:
            unresolved.append(
                f"scheduled debit {event.source_event_id} has no settlement date"
            )
            continue
        if not _in_horizon(cash_date, horizon_start, horizon_end):
            continue
        if event.amount_home_currency is None:
            unresolved.append(
                f"scheduled debit {event.source_event_id} has an unresolved amount "
                "and cannot be treated as zero"
            )
            continue
        amount = event.amount_home_currency
        override = override_for_event(
            event, state.recurring_series_candidates, overrides
        )
        if override is not None:
            if override.new_amount is None:
                continue
            amount = override.new_amount
        entries.append(
            ForecastCashEvent(
                date=cash_date,
                amount_home_currency=amount,
                direction=EventDirection.DEBIT,
                signed_amount=_signed(EventDirection.DEBIT, amount),
                kind=ForecastEventKind.SCHEDULED_DEBIT,
                source=ForecastEventSource.EXPLICIT_EVENT,
                source_event_id=event.source_event_id,
                source_series_key=event_series_key(event),
                is_generated_recurrence=False,
                priority=SameDayPriority.OBLIGATED_DEBIT,
                description=f"scheduled debit {event.source_event_id} ({event.description})",
                original_event_date=event.event_date,
                original_settlement_date=event.settlement_date,
                original_cash_date=event.cash_date,
                flexibility=event.flexibility,
                category=event.category,
                order_key=event.source_event_id,
            )
        )
    return entries


def _confirmed_credit_entries(
    state: NormalizedFinancialState,
    horizon_start: date,
    horizon_end: date,
    unresolved: list[str],
) -> list[ForecastCashEvent]:
    entries: list[ForecastCashEvent] = []
    for event in state.confirmed_scheduled_income:
        cash_date = event.cash_date or event.settlement_date
        if cash_date is None:
            unresolved.append(
                f"confirmed income {event.source_event_id} has no settlement date"
            )
            continue
        if not _in_horizon(cash_date, horizon_start, horizon_end):
            continue
        if event.amount_home_currency is None:
            unresolved.append(
                f"confirmed income {event.source_event_id} has an unresolved amount "
                "and cannot be treated as zero"
            )
            continue
        entries.append(
            ForecastCashEvent(
                date=cash_date,
                amount_home_currency=event.amount_home_currency,
                direction=EventDirection.CREDIT,
                signed_amount=_signed(EventDirection.CREDIT, event.amount_home_currency),
                kind=ForecastEventKind.CONFIRMED_CREDIT,
                source=ForecastEventSource.EXPLICIT_EVENT,
                source_event_id=event.source_event_id,
                source_series_key=event_series_key(event),
                is_generated_recurrence=False,
                priority=SameDayPriority.CONFIRMED_CREDIT,
                description=(
                    f"confirmed scheduled credit {event.source_event_id} "
                    f"({event.description})"
                ),
                original_event_date=event.event_date,
                original_settlement_date=event.settlement_date,
                original_cash_date=event.cash_date,
                flexibility=event.flexibility,
                category=event.category,
                order_key=event.source_event_id,
                confirmation_type="confirmed",
                evidence_source_ids=(event.source_event_id,),
                generated_from_history=False,
                explicitly_scheduled=True,
                message_confirmed=False,
                income_subtype=classify_income_text(event.description, event.category).value,
            )
        )
    return entries


def _adjustments(state: NormalizedFinancialState) -> tuple[ForecastAdjustment, ...]:
    return getattr(state, "forecast_adjustments", ()) or ()


def _has_adjustment(state: NormalizedFinancialState, kind: ForecastAdjustmentKind) -> bool:
    return any(item.kind is kind for item in _adjustments(state))


def _salary_amount_for_date(
    state: NormalizedFinancialState,
    when: date,
    baseline: Decimal,
    salary_index: int,
    series: RecurringSeriesCandidate,
) -> Decimal:
    amount = baseline
    candidates = state.recurring_series_candidates
    for item in _adjustments(state):
        if not amendment_targets_series(item, series, candidates=candidates):
            continue
        if item.kind is ForecastAdjustmentKind.SALARY_AMOUNT and item.amount_home is not None:
            effective = item.effective_date or state.request_date
            if when >= effective:
                amount = item.amount_home
        if item.kind is ForecastAdjustmentKind.SALARY_TEMPORARY and item.amount_home is not None:
            allowed = item.occurrences if item.occurrences is not None else 1
            if salary_index < allowed:
                amount = item.amount_home
    return amount


def _adjustment_entries(
    state: NormalizedFinancialState,
    horizon_start: date,
    horizon_end: date,
    existing: list[ForecastCashEvent],
    config: ForecastConfig,
) -> list[ForecastCashEvent]:
    occupied_dates = {
        entry.date
        for entry in existing
        if entry.category == "salary" and entry.direction is EventDirection.CREDIT
    }
    extras: list[ForecastCashEvent] = []
    for item in _adjustments(state):
        if item.kind not in {
            ForecastAdjustmentKind.START_SALARY,
            ForecastAdjustmentKind.CONFIRM_INCOME,
        }:
            continue
        if item.amount_home is None or item.effective_date is None:
            continue
        if not _in_horizon(item.effective_date, horizon_start, horizon_end):
            continue
        if item.kind is ForecastAdjustmentKind.START_SALARY and item.effective_date in occupied_dates:
            continue
        dates = [item.effective_date]
        if item.kind is ForecastAdjustmentKind.START_SALARY and continue_start_salary(
            config.salary_projection_mode
        ):
            dates.extend(
                iter_recurrence_dates(
                    last_observed=item.effective_date,
                    cadence=Cadence.MONTHLY,
                    horizon_start=horizon_start,
                    horizon_end=horizon_end,
                    pattern_day=item.effective_date.day,
                )
            )
        for when in dates:
            if not _in_horizon(when, horizon_start, horizon_end):
                continue
            if when in occupied_dates and item.kind is ForecastAdjustmentKind.START_SALARY:
                continue
            extras.append(
                ForecastCashEvent(
                    date=when,
                    amount_home_currency=item.amount_home,
                    direction=EventDirection.CREDIT,
                    signed_amount=_signed(EventDirection.CREDIT, item.amount_home),
                    kind=(
                        ForecastEventKind.GENERATED_RECURRING_CREDIT
                        if when != item.effective_date
                        else ForecastEventKind.CONFIRMED_CREDIT
                    ),
                    source=(
                        ForecastEventSource.GENERATED_RECURRENCE
                        if when != item.effective_date
                        else ForecastEventSource.EXPLICIT_EVENT
                    ),
                    source_event_id=f"evidence:{','.join(item.source_ids)}:{when.isoformat()}",
                    source_series_key=None,
                    is_generated_recurrence=when != item.effective_date,
                    priority=SameDayPriority.CONFIRMED_CREDIT,
                    description=f"evidence-confirmed credit ({item.notes})",
                    category=item.category or "income",
                    order_key=f"evidence:{','.join(item.source_ids)}:{when.isoformat()}",
                    confirmation_type=(
                        "continuation_supported"
                        if when != item.effective_date
                        else "confirmed"
                    ),
                    evidence_source_ids=item.source_ids,
                    generated_from_history=False,
                    explicitly_scheduled=False,
                    message_confirmed=True,
                    income_subtype=item.income_subtype or (
                        "base_salary" if item.category == "salary" else "other"
                    ),
                )
            )
            occupied_dates.add(when)
    return extras


def _generated_entries(
    state: NormalizedFinancialState,
    config: ForecastConfig,
    horizon_start: date,
    horizon_end: date,
    occupied: tuple[NormalizedCashEvent, ...],
    unresolved: list[str],
    message_flags: list[str],
    overrides: tuple[RecurrenceOverride, ...] = (),
) -> list[ForecastCashEvent]:
    entries: list[ForecastCashEvent] = []
    payday = next(
        (item for item in _adjustments(state) if item.kind is ForecastAdjustmentKind.SALARY_PAYDAY),
        None,
    )
    salary_index_by_key: dict[str, int] = {}
    for series in state.recurring_series_candidates:
        is_salary = _is_salary_series(series)
        debit_override = (
            override_for_series(series, overrides)
            if series.direction is EventDirection.DEBIT
            else None
        )
        if debit_override is not None and debit_override.new_amount is None:
            continue
        if is_salary:
            eligible, income_confidence, _reason = salary_generation_allowed(
                series,
                mode=config.salary_projection_mode,
                adjustments=_adjustments(state),
                candidates=state.recurring_series_candidates,
            )
        elif series.direction is EventDirection.CREDIT:
            eligible = False
            income_confidence = None
        else:
            eligible = _eligible_for_auto_projection(series, config)
            income_confidence = None
        if not eligible:
            continue
        if not series.observed_dates:
            continue
        amount = _projected_amount(series, config)
        if debit_override is not None and debit_override.new_amount is not None:
            amount = debit_override.new_amount
        if amount is None:
            unresolved.append(
                f"recurring series {series_key(series)} has no usable amount "
                "and cannot be projected as zero"
            )
            continue
        last_observed = max(series.observed_dates)
        pattern_day = last_observed.day
        cadence = series.inferred_cadence
        if (
            is_salary
            and payday is not None
            and payday.effective_date is not None
            and amendment_targets_series(
                payday, series, candidates=state.recurring_series_candidates
            )
        ):
            pattern_day = payday.effective_date.day
            last_observed = add_calendar_months(payday.effective_date, -1, pattern_day=pattern_day)
            cadence = Cadence.MONTHLY
        key = series_key(series)
        if series.amount_behavior.value == "fixed" and series.representative_amount is not None:
            strategy_name = "fixed"
        elif series.direction is EventDirection.CREDIT:
            strategy_name = VariableAmountStrategy.LATEST.value
        else:
            strategy_name = config.variable_amount_strategy.value
        flagged = series.requires_message_resolution and not _adjustments(state)
        if flagged:
            message_flags.append(
                f"{series.category} series {series.original_description!r} may be "
                f"affected by unread messages ({', '.join(state.user_message_ids) or 'none'}); "
                "projected from historical baseline only — no raise, cut, or "
                "employment end was assumed"
            )
        message_confirmed = is_salary and any(
            amendment_targets_series(
                item, series, candidates=state.recurring_series_candidates
            )
            for item in _adjustments(state)
            if item.kind
            in {
                ForecastAdjustmentKind.SALARY_AMOUNT,
                ForecastAdjustmentKind.SALARY_TEMPORARY,
                ForecastAdjustmentKind.SALARY_PAYDAY,
                ForecastAdjustmentKind.START_SALARY,
            }
        )
        evidence_ids = tuple(
            source
            for item in _adjustments(state)
            if amendment_targets_series(
                item, series, candidates=state.recurring_series_candidates
            )
            for source in item.source_ids
        )
        if series.scheduled_confirmed_event_ids:
            evidence_ids = series.scheduled_confirmed_event_ids + evidence_ids
        confirmation_type = (
            income_confidence.value if income_confidence is not None else ""
        )
        generated_from_history = is_salary and not message_confirmed and not bool(
            series.scheduled_confirmed_event_ids
        )
        provenance = RecurrenceProvenance(
            series_key=key,
            source_event_ids=series.event_ids,
            generation_method="calendar_cadence",
            cadence=series.inferred_cadence,
            amount_strategy=strategy_name,
            confidence=series.cadence_confidence,
            requires_message_confirmation=flagged,
            confirmation_type=confirmation_type,
            evidence_source_ids=evidence_ids,
            generated_from_history=generated_from_history,
            explicitly_scheduled=bool(series.scheduled_confirmed_event_ids),
            message_confirmed=message_confirmed,
            income_subtype=classify_series(series).value if is_salary else "",
        )
        for when in iter_recurrence_dates(
            last_observed=last_observed,
            cadence=cadence,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            pattern_day=pattern_day,
            cadence_days=series.inferred_cadence_days,
        ):
            if _is_duplicate_of_explicit(
                series=series,
                generated_date=when,
                occupied=occupied,
            ):
                continue
            projected = amount
            if is_salary:
                key_index = series_key(series)
                salary_index = salary_index_by_key.get(key_index, 0)
                projected = _salary_amount_for_date(
                    state, when, amount, salary_index, series
                )
                salary_index_by_key[key_index] = salary_index + 1
                kind = ForecastEventKind.GENERATED_RECURRING_CREDIT
                priority = SameDayPriority.CONFIRMED_CREDIT
                direction = EventDirection.CREDIT
                label = "generated recurring salary"
            else:
                kind = ForecastEventKind.GENERATED_RECURRING_DEBIT
                priority = SameDayPriority.GENERATED_RECURRING_DEBIT
                direction = EventDirection.DEBIT
                label = "generated recurring debit"
                for item in _adjustments(state):
                    if item.kind is not ForecastAdjustmentKind.RENT_PERCENT:
                        continue
                    if (item.category or "rent") != series.category:
                        continue
                    if item.percent is None:
                        continue
                    effective = item.effective_date or state.request_date
                    if when >= effective:
                        projected = (projected * (Decimal("1") + item.percent / Decimal("100"))).quantize(
                            Decimal("0.01")
                        )
            generated_id = f"generated:{key}:{when.isoformat()}"
            entries.append(
                ForecastCashEvent(
                    date=when,
                    amount_home_currency=projected,
                    direction=direction,
                    signed_amount=_signed(direction, projected),
                    kind=kind,
                    source=ForecastEventSource.GENERATED_RECURRENCE,
                    source_event_id=generated_id,
                    source_series_key=key,
                    is_generated_recurrence=True,
                    priority=priority,
                    description=(
                        f"{label} {series.category} {series.original_description!r} "
                        f"{series.inferred_cadence.value} {series.cadence_confidence.value}"
                    ),
                    provenance=provenance,
                    flexibility=series.flexibility,
                    category=series.category,
                    requires_message_confirmation=flagged,
                    order_key=generated_id,
                    confirmation_type=confirmation_type,
                    evidence_source_ids=evidence_ids,
                    generated_from_history=generated_from_history,
                    explicitly_scheduled=bool(series.scheduled_confirmed_event_ids),
                    message_confirmed=message_confirmed,
                    income_subtype=classify_series(series).value if is_salary else "",
                )
            )
    return entries


def _collect_unresolved_diagnostics(
    state: NormalizedFinancialState,
    unresolved: list[str],
) -> None:
    for event in state.unresolved_events:
        if event.amount_home_currency is not None:
            continue
        if event.is_prospective or event.requires_external_evidence:
            reason = (
                f"event {event.source_event_id} requires external evidence"
                if event.requires_external_evidence
                else f"event {event.source_event_id} has an unresolved amount"
            )
            if event.requires_external_evidence:
                reason += " (blank amount; image not read)"
            if reason not in unresolved:
                unresolved.append(reason)
    if state.requires_message_resolution and not any(
        series.category == "salary" and series.requires_message_resolution
        for series in state.recurring_series_candidates
    ):
        # Messages exist but no salary series claimed them. Still surface them.
        pass


def _ignored_summaries(state: NormalizedFinancialState) -> tuple[str, ...]:
    lines: list[str] = []
    for event in state.ignored_pending_credits:
        amount = (
            f"+{event.amount_home_currency}"
            if event.amount_home_currency is not None
            else "unresolved"
        )
        lines.append(
            f"pending credit {event.source_event_id} {amount} ignored "
            f"({event.description})"
        )
    for event in state.ignored_events:
        if event.ignore_reason is None:
            continue
        if event.ignore_reason.value == "pending_credit":
            continue
        if event.ignore_reason.value == "unconfirmed_scheduled_credit":
            lines.append(
                f"unconfirmed scheduled credit {event.source_event_id} ignored "
                f"({event.description})"
            )
    return tuple(lines)


def _generated_summaries(entries: list[ForecastCashEvent]) -> tuple[str, ...]:
    seen: dict[str, list[ForecastCashEvent]] = defaultdict(list)
    for entry in entries:
        if not entry.is_generated_recurrence or entry.source_series_key is None:
            continue
        seen[entry.source_series_key].append(entry)
    summaries: list[str] = []
    for key, group in seen.items():
        sample = group[0]
        cadence = sample.provenance.cadence.value if sample.provenance else "unknown"
        confidence = (
            sample.provenance.confidence.value if sample.provenance else "unknown"
        )
        summaries.append(
            f"{sample.category or 'series'} {cadence} {confidence} "
            f"n={len(group)} confirm={sample.confirmation_type or 'n/a'} "
            f"subtype={sample.income_subtype or 'n/a'} ({sample.description})"
        )
    return tuple(summaries)


def apply_forecast_events(
    *,
    opening_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    entries: tuple[ForecastCashEvent, ...],
    horizon_start: date,
) -> tuple[
    tuple[DailyForecast, ...],
    Decimal,
    date | None,
    ForecastCashEvent | None,
    bool,
]:
    """Apply ordered events and check the floor after every event."""
    by_day: dict[date, list[ForecastCashEvent]] = defaultdict(list)
    for entry in entries:
        by_day[entry.date].append(entry)

    balance = opening_balance
    minimum_observed = opening_balance
    first_violation_date: date | None = None
    first_violation_entry: ForecastCashEvent | None = None
    if balance < minimum_balance_to_keep:
        first_violation_date = horizon_start
        first_violation_entry = None

    daily: list[DailyForecast] = []
    for day in sorted(by_day):
        day_entries = tuple(sorted(by_day[day], key=_sort_key))
        opening = balance
        running: list[Decimal] = []
        inflows = _ZERO
        outflows = _ZERO
        day_min = opening
        violated = False
        for entry in day_entries:
            balance += entry.signed_amount
            running.append(balance)
            if entry.signed_amount > 0:
                inflows += entry.signed_amount
            elif entry.signed_amount < 0:
                outflows += -entry.signed_amount
            if balance < day_min:
                day_min = balance
            if balance < minimum_observed:
                minimum_observed = balance
            if balance < minimum_balance_to_keep:
                violated = True
                if first_violation_date is None:
                    first_violation_date = day
                    first_violation_entry = entry
        daily.append(
            DailyForecast(
                date=day,
                opening_balance=opening,
                inflows=inflows,
                outflows=outflows,
                closing_balance=balance,
                minimum_balance=day_min,
                violated_minimum=violated,
                entries=day_entries,
                running_balances=tuple(running),
            )
        )

    is_safe = first_violation_date is None
    return tuple(daily), minimum_observed, first_violation_date, first_violation_entry, is_safe


def forecast_financial_state(
    normalized_state: NormalizedFinancialState,
    strategy_config: ForecastConfig | None = None,
    *,
    candidate_payments: tuple[ForecastCashEvent, ...] = (),
    recurrence_overrides: tuple[RecurrenceOverride, ...] = (),
) -> ForecastResult:
    """Simulate cash from request_date through request_date + horizon_days.

    Horizon is inclusive on both ends. Historical settled events are evidence
    only and are never replayed onto the opening snapshot.

    `candidate_payments` is an extension point for Phase 4. Phase 3 never
    creates candidate payments; callers may inject them only in tests.

    `recurrence_overrides` apply only to future expense recurrences and
    matching future scheduled debits. Pending authorized debits and
    category-level essential reserves are unchanged. The baseline state is
    not mutated.
    """
    config = strategy_config or ForecastConfig()
    state = normalized_state
    horizon_start = state.request_date
    horizon_end = config.horizon_end(horizon_start)
    unresolved: list[str] = []
    message_flags: list[str] = []

    entries: list[ForecastCashEvent] = []
    entries.extend(_pending_debit_entries(state, config, horizon_start, unresolved))
    entries.extend(
        _scheduled_debit_entries(
            state, horizon_start, horizon_end, unresolved, recurrence_overrides
        )
    )
    entries.extend(
        _confirmed_credit_entries(state, horizon_start, horizon_end, unresolved)
    )

    occupied = _explicit_occupancy(state)
    entries.extend(
        _generated_entries(
            state,
            config,
            horizon_start,
            horizon_end,
            occupied,
            unresolved,
            message_flags,
            recurrence_overrides,
        )
    )
    entries.extend(_adjustment_entries(state, horizon_start, horizon_end, entries, config))
    entries.extend(essential_spend_entries(state, config, tuple(entries)))

    for candidate in candidate_payments:
        if candidate.priority is not SameDayPriority.CANDIDATE_PAYMENT:
            raise ValueError("candidate payments must use SameDayPriority.CANDIDATE_PAYMENT")
        if not _in_horizon(candidate.date, horizon_start, horizon_end):
            continue
        entries.append(candidate)

    _collect_unresolved_diagnostics(state, unresolved)
    if (
        state.requires_message_resolution
        and any(_is_salary_series(series) for series in state.recurring_series_candidates)
        and not message_flags
        and not _adjustments(state)
    ):
        message_flags.append(
            "unread messages exist for this user; salary continuation was not "
            f"interpreted ({', '.join(state.user_message_ids) or 'none'})"
        )

    unique_unresolved = tuple(dict.fromkeys(unresolved))
    unique_flags = tuple(dict.fromkeys(message_flags))
    if config.strict_unresolved_amounts and unique_unresolved:
        raise UnresolvedForecastError(unique_unresolved)

    ordered = tuple(sorted(entries, key=lambda item: (item.date, *_sort_key(item))))
    daily, minimum_observed, first_date, first_entry, is_safe = apply_forecast_events(
        opening_balance=state.opening_balance,
        minimum_balance_to_keep=state.profile.minimum_balance_to_keep,
        entries=ordered,
        horizon_start=horizon_start,
    )

    return ForecastResult(
        request_id=state.request_id,
        opening_balance=state.opening_balance,
        minimum_balance_to_keep=state.profile.minimum_balance_to_keep,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        daily_forecasts=daily,
        all_entries=ordered,
        minimum_observed_balance=minimum_observed,
        first_violation_date=first_date,
        first_violation_entry=first_entry,
        is_safe=is_safe,
        unresolved_reasons=unique_unresolved,
        generated_recurrence_summaries=_generated_summaries(entries),
        ignored_summaries=_ignored_summaries(state),
        message_uncertainty_flags=unique_flags,
    )
