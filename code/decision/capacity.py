"""Deterministic safe-payment capacity on the evidence-resolved forecast.

Computes amount_safe_to_pay and earliest_date_for_full_payment before optional
spending changes and independently of payment-method preferences.

Assumption (monotonicity): if paying X on a fixed date is unsafe, any payment
greater than X on that same date is also unsafe. Candidate payments are applied
last on their day (SameDayPriority.CANDIDATE_PAYMENT = 40), so they cannot
unlock later credits.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import ROUND_DOWN, Decimal

from data.models import EventDirection, FinanceRequest
from decision.models import CapacityResult, PaymentSafety
from finance.forecast import forecast_financial_state
from finance.forecast_models import (
    ForecastCashEvent,
    ForecastConfig,
    ForecastEventKind,
    ForecastEventSource,
    ForecastResult,
    SameDayPriority,
)
from finance.models import NormalizedFinancialState

_ZERO = Decimal("0")


def infer_money_quantum(*amounts: Decimal) -> Decimal:
    """Smallest decimal place present on the supplied monetary values.

    Integer-only amounts use quantum 1. Mixed cent values use 0.01 (or finer
    if a source amount actually has a finer exponent). No currency table.
    """
    exponents: list[int] = []
    for amount in amounts:
        exponent = amount.as_tuple().exponent
        if isinstance(exponent, int):
            exponents.append(exponent)
    if not exponents:
        return Decimal("0.01")
    minimum = min(exponents)
    if minimum >= 0:
        return Decimal("1")
    return Decimal("1").scaleb(minimum)


def _quantize_down(amount: Decimal, quantum: Decimal) -> Decimal:
    if amount <= _ZERO:
        return _ZERO
    units = (amount / quantum).to_integral_value(rounding=ROUND_DOWN)
    return units * quantum


def candidate_payment(
    request: FinanceRequest,
    payment_date: date,
    amount: Decimal,
) -> ForecastCashEvent:
    return ForecastCashEvent(
        date=payment_date,
        amount_home_currency=amount,
        direction=EventDirection.DEBIT,
        signed_amount=-amount,
        kind=ForecastEventKind.CANDIDATE_PAYMENT,
        source=ForecastEventSource.CANDIDATE,
        source_event_id=f"candidate:{request.request_id}",
        source_series_key=None,
        is_generated_recurrence=False,
        priority=SameDayPriority.CANDIDATE_PAYMENT,
        description=f"candidate payment {request.request_id}",
        order_key=f"candidate:{request.request_id}",
    )


def is_payment_safe(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    payment_date: date,
    amount: Decimal,
    config: ForecastConfig | None = None,
) -> PaymentSafety:
    """Inject one candidate payment and reuse the Phase 3 simulator."""
    strategy = config or ForecastConfig(strict_unresolved_amounts=True)
    payments = () if amount == _ZERO else (candidate_payment(request, payment_date, amount),)
    forecast = forecast_financial_state(
        state,
        strategy,
        candidate_payments=payments,
    )
    return PaymentSafety(
        payment_date=payment_date,
        amount=amount,
        is_safe=forecast.is_safe,
        forecast=forecast,
        minimum_observed_balance=forecast.minimum_observed_balance,
        first_violation_date=forecast.first_violation_date,
        first_violation_entry=forecast.first_violation_entry,
    )


def _balances_reduced_by_payment(
    baseline: ForecastResult,
    payment_date: date,
) -> tuple[Decimal, ...]:
    """Balances that fall by the payment amount after same-day priority 40."""
    last_close = baseline.opening_balance
    reduced: list[Decimal] = []
    saw_day = False
    for day in baseline.daily_forecasts:
        if day.date < payment_date:
            last_close = day.closing_balance
            continue
        if day.date == payment_date:
            saw_day = True
            reduced.append(day.closing_balance)
            last_close = day.closing_balance
            continue
        reduced.extend(day.running_balances)
        last_close = day.closing_balance
    if not saw_day:
        reduced.insert(0, last_close)
    if not reduced:
        reduced.append(last_close)
    return tuple(reduced)


def _tightest_constraint(
    baseline: ForecastResult,
    payment_date: date,
    minimum_balance_to_keep: Decimal,
) -> tuple[date | None, ForecastCashEvent | None]:
    last_close = baseline.opening_balance
    last_entry: ForecastCashEvent | None = None
    last_date = baseline.horizon_start
    best_slack: Decimal | None = None
    limiting_date: date | None = None
    limiting_entry: ForecastCashEvent | None = None

    def consider(balance: Decimal, when: date, entry: ForecastCashEvent | None) -> None:
        nonlocal best_slack, limiting_date, limiting_entry
        slack = balance - minimum_balance_to_keep
        if best_slack is None or slack < best_slack:
            best_slack = slack
            limiting_date = when
            limiting_entry = entry

    saw_day = False
    for day in baseline.daily_forecasts:
        if day.date < payment_date:
            last_close = day.closing_balance
            last_date = day.date
            if day.entries:
                last_entry = day.entries[-1]
            continue
        if day.date == payment_date:
            saw_day = True
            consider(day.closing_balance, day.date, day.entries[-1] if day.entries else None)
            last_close = day.closing_balance
            continue
        for entry, balance in zip(day.entries, day.running_balances):
            consider(balance, day.date, entry)
        last_close = day.closing_balance
    if not saw_day:
        consider(last_close, payment_date if last_date <= payment_date else last_date, last_entry)
    return limiting_date, limiting_entry


def derived_headroom(
    baseline: ForecastResult,
    payment_date: date,
    requested_amount: Decimal,
    quantum: Decimal,
) -> Decimal:
    """Exact same-day-last-payment headroom from the baseline path."""
    min_keep = baseline.minimum_balance_to_keep
    slack = min(
        balance - min_keep
        for balance in _balances_reduced_by_payment(baseline, payment_date)
    )
    if slack <= _ZERO:
        return _ZERO
    return _quantize_down(min(requested_amount, slack), quantum)


def _binary_search_safe_amount(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    payment_date: date,
    quantum: Decimal,
    config: ForecastConfig,
) -> Decimal:
    requested = request.requested_amount
    if not is_payment_safe(state, request, payment_date, _ZERO, config).is_safe:
        return _ZERO
    if is_payment_safe(state, request, payment_date, requested, config).is_safe:
        return requested
    high_units = int((requested / quantum).to_integral_value(rounding=ROUND_DOWN))
    low_units = 0
    while low_units < high_units:
        mid_units = (low_units + high_units + 1) // 2
        mid = quantum * mid_units
        if is_payment_safe(state, request, payment_date, mid, config).is_safe:
            low_units = mid_units
        else:
            high_units = mid_units - 1
    return quantum * low_units


def _verified_safe_amount(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    payment_date: date,
    candidate: Decimal,
    quantum: Decimal,
    config: ForecastConfig,
) -> Decimal:
    requested = request.requested_amount
    amount = min(requested, max(_ZERO, candidate))
    amount = _quantize_down(amount, quantum)
    if amount > _ZERO and not is_payment_safe(
        state, request, payment_date, amount, config
    ).is_safe:
        return _binary_search_safe_amount(state, request, payment_date, quantum, config)
    if amount < requested:
        step = amount + quantum
        if step <= requested and is_payment_safe(
            state, request, payment_date, step, config
        ).is_safe:
            return _binary_search_safe_amount(state, request, payment_date, quantum, config)
    return amount


def amount_safe_to_pay(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    config: ForecastConfig | None = None,
    *,
    baseline: ForecastResult | None = None,
) -> Decimal:
    strategy = config or ForecastConfig(strict_unresolved_amounts=True)
    base = baseline or forecast_financial_state(state, strategy)
    quantum = infer_money_quantum(
        request.requested_amount,
        state.opening_balance,
        state.profile.minimum_balance_to_keep,
    )
    derived = derived_headroom(base, request.request_date, request.requested_amount, quantum)
    return _verified_safe_amount(
        state, request, request.request_date, derived, quantum, strategy
    )


def earliest_date_for_full_payment(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    config: ForecastConfig | None = None,
    *,
    baseline: ForecastResult | None = None,
) -> date | None:
    strategy = config or ForecastConfig(strict_unresolved_amounts=True)
    base = baseline or forecast_financial_state(state, strategy)
    quantum = infer_money_quantum(
        request.requested_amount,
        state.opening_balance,
        state.profile.minimum_balance_to_keep,
    )
    cursor = request.request_date
    while cursor <= base.horizon_end:
        headroom = derived_headroom(base, cursor, request.requested_amount, quantum)
        if headroom >= request.requested_amount:
            checked = is_payment_safe(
                state, request, cursor, request.requested_amount, strategy
            )
            if checked.is_safe:
                return cursor
        cursor += timedelta(days=1)
    return None


def compute_capacity(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    config: ForecastConfig | None = None,
) -> CapacityResult:
    """Capacity on the supplied (evidence-resolved) state. Does not read preferences."""
    strategy = config or ForecastConfig(strict_unresolved_amounts=True)
    baseline = forecast_financial_state(state, strategy)
    quantum = infer_money_quantum(
        request.requested_amount,
        state.opening_balance,
        state.profile.minimum_balance_to_keep,
    )
    safe_today = amount_safe_to_pay(state, request, strategy, baseline=baseline)
    today_forecast = is_payment_safe(
        state, request, request.request_date, safe_today, strategy
    )
    full_today = (
        safe_today == request.requested_amount and today_forecast.is_safe
    )
    if full_today:
        earliest = request.request_date
    else:
        earliest = earliest_date_for_full_payment(
            state, request, strategy, baseline=baseline
        )
    limiting_date, limiting_event = _tightest_constraint(
        baseline, request.request_date, baseline.minimum_balance_to_keep
    )
    unsafe_min: Decimal | None = None
    bump = safe_today + quantum
    if safe_today < request.requested_amount:
        unsafe = is_payment_safe(state, request, request.request_date, bump, strategy)
        unsafe_min = unsafe.minimum_observed_balance
    return CapacityResult(
        request_id=request.request_id,
        amount_safe_to_pay=safe_today,
        earliest_date_for_full_payment=earliest,
        full_payment_safe_today=full_today,
        baseline_minimum_balance=baseline.minimum_observed_balance,
        safe_payment_forecast_minimum=today_forecast.minimum_observed_balance,
        limiting_date=limiting_date,
        limiting_event=limiting_event,
        unresolved_reasons=baseline.unresolved_reasons,
        quantum=quantum,
        requested_amount=request.requested_amount,
        request_date=request.request_date,
        unsafe_increment_minimum=unsafe_min,
        baseline=baseline,
        safe_today_forecast=today_forecast.forecast,
    )
