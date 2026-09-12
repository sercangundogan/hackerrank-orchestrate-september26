"""Central deterministic verifier for every payment candidate.

No plan may be recommended unless this module accepts it. Capacity numbers
are inputs; this module does not recompute amount_safe_to_pay.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from data.models import (
    AffordabilityStatus,
    ConsideredPaymentMethod,
    EventDirection,
    FinanceRequest,
    Flexibility,
    PaymentOption,
    PaymentOptionMethod,
    RecommendedPaymentMethod,
)
from decision.capacity import candidate_schedule
from decision.models import CandidatePlan, SpendingAction, SpendingActionKind
from decision.plans import considers, installment_accepted, installment_schedule
from finance.forecast import forecast_financial_state
from finance.forecast_models import ForecastConfig, RecurrenceOverride
from finance.models import NormalizedFinancialState

_ZERO = Decimal("0")
_FLEXIBLE = {
    Flexibility.REDUCIBLE,
    Flexibility.STOPPABLE,
    Flexibility.REDUCIBLE_OR_STOPPABLE,
}


def overrides_from_actions(actions: tuple[SpendingAction, ...]) -> tuple[RecurrenceOverride, ...]:
    return tuple(
        RecurrenceOverride(event_id=action.event_id, new_amount=action.new_amount)
        for action in actions
    )


def simulate_plan(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    plan: CandidatePlan,
    config: ForecastConfig | None = None,
):
    strategy = config or ForecastConfig(strict_unresolved_amounts=True)
    payments = tuple((item.date, item.amount) for item in plan.payments)
    return forecast_financial_state(
        state,
        strategy,
        candidate_payments=candidate_schedule(request, payments),
        recurrence_overrides=overrides_from_actions(plan.spending_changes),
    )


def _spending_failures(
    state: NormalizedFinancialState,
    actions: tuple[SpendingAction, ...],
) -> list[str]:
    failures: list[str] = []
    if len(actions) > 3:
        failures.append("more_than_three_spending_changes")
    seen: set[str] = set()
    protected = set(state.profile.expense_categories_to_protect)
    reduce_ok = set(state.profile.expense_categories_user_is_willing_to_reduce)
    stop_ok = set(state.profile.expense_categories_user_is_willing_to_stop)
    series_by_id = {
        event_id: series
        for series in state.recurring_series_candidates
        for event_id in series.event_ids
    }
    for action in actions:
        if action.event_id in seen:
            failures.append(f"duplicate_action_on_{action.event_id}")
        seen.add(action.event_id)
        series = series_by_id.get(action.event_id)
        if series is None:
            failures.append(f"unknown_series_{action.event_id}")
            continue
        if series.direction is not EventDirection.DEBIT:
            failures.append(f"cannot_change_income_{action.event_id}")
        if series.flexibility not in _FLEXIBLE:
            failures.append(f"fixed_event_{action.event_id}")
        if series.is_protected_category or series.category in protected:
            failures.append(f"protected_category_{action.event_id}")
        if action.kind is SpendingActionKind.STOP:
            if series.flexibility not in {
                Flexibility.STOPPABLE,
                Flexibility.REDUCIBLE_OR_STOPPABLE,
            }:
                failures.append(f"not_stoppable_{action.event_id}")
            if action.category not in stop_ok and not series.profile_allows_stop:
                failures.append(f"stop_not_permitted_{action.event_id}")
            if action.new_amount is not None:
                failures.append(f"stop_must_zero_{action.event_id}")
        else:
            if series.flexibility not in {
                Flexibility.REDUCIBLE,
                Flexibility.REDUCIBLE_OR_STOPPABLE,
            }:
                failures.append(f"not_reducible_{action.event_id}")
            if action.category not in reduce_ok and not series.profile_allows_reduce:
                failures.append(f"reduce_not_permitted_{action.event_id}")
            if series.minimum_allowed_amount is None or action.new_amount is None:
                failures.append(f"reduce_missing_minimum_{action.event_id}")
            elif action.new_amount < series.minimum_allowed_amount:
                failures.append(f"reduce_below_minimum_{action.event_id}")
    return failures


def validate_candidate(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    plan: CandidatePlan,
    options: tuple[PaymentOption, ...],
    amount_safe: Decimal,
    earliest: date | None,
    config: ForecastConfig | None = None,
) -> CandidatePlan:
    """Return a copy of `plan` with safety, preference, and failure fields set."""
    failures: list[str] = []
    preference = True
    horizon_end = (config or ForecastConfig()).horizon_end(request.request_date)

    if plan.method is RecommendedPaymentMethod.NOT_RECOMMENDED:
        if plan.payments:
            failures.append("not_recommended_must_have_no_payments")
        if plan.spending_changes:
            failures.append("not_recommended_must_have_no_spending_changes")
        return CandidatePlan(
            method=plan.method,
            affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
            payments=plan.payments,
            payment_option_id=plan.payment_option_id,
            spending_changes=plan.spending_changes,
            total_paid=plan.total_paid,
            start_date=plan.start_date,
            completion_date=plan.completion_date,
            completes_by_deadline=False,
            requires_spending_changes=False,
            is_preference_eligible=True,
            is_financially_safe=not failures,
            validation_failures=tuple(failures),
            forecast_result=None,
        )

    if plan.method is RecommendedPaymentMethod.FULL_PAYMENT:
        preference = considers(state, ConsideredPaymentMethod.FULL_PAYMENT)
        if len(plan.payments) != 1:
            failures.append("full_requires_one_payment")
        elif plan.payments[0].date != request.request_date:
            failures.append("full_must_be_on_request_date")
        if plan.total_paid != request.requested_amount:
            failures.append("full_total_must_equal_requested")
        if plan.affordability_status is AffordabilityStatus.AFFORDABLE_NOW and plan.spending_changes:
            failures.append("full_with_changes_is_not_affordable_now")

    elif plan.method is RecommendedPaymentMethod.PARTIAL_PAYMENT:
        preference = considers(state, ConsideredPaymentMethod.PARTIAL_PAYMENT)
        if not request.allows_partial_payment:
            failures.append("request_disallows_partial")
        if not (_ZERO < amount_safe < request.requested_amount):
            failures.append("partial_requires_strict_partial_safe_amount")
        if earliest is None:
            failures.append("partial_requires_earliest_full_date")
        elif earliest > request.desired_completion_date:
            failures.append("partial_second_payment_after_deadline")
        if len(plan.payments) != 2:
            failures.append("partial_requires_exactly_two_payments")
        else:
            first, second = plan.payments
            if first.date != request.request_date:
                failures.append("partial_first_payment_not_request_date")
            if first.amount != amount_safe:
                failures.append("partial_first_amount_must_equal_amount_safe")
            if earliest is not None and second.date != earliest:
                failures.append("partial_second_date_must_equal_earliest")
            if second.amount != request.requested_amount - amount_safe:
                failures.append("partial_second_amount_incorrect")
        if plan.total_paid != request.requested_amount:
            failures.append("partial_total_must_equal_requested")

    elif plan.method is RecommendedPaymentMethod.INSTALLMENTS:
        preference = considers(state, ConsideredPaymentMethod.INSTALLMENTS)
        if state.profile.max_installment_months is None:
            failures.append("installments_max_months_blank")
        if plan.payment_option_id is None:
            failures.append("installments_require_seller_option")
        else:
            option = next(
                (item for item in options if item.payment_option_id == plan.payment_option_id),
                None,
            )
            if option is None:
                failures.append("unknown_payment_option")
            elif option.payment_method is not PaymentOptionMethod.INSTALLMENTS:
                failures.append("option_is_not_installments")
            else:
                if not installment_accepted(state, option):
                    failures.append("exceeds_max_installment_months")
                expected = installment_schedule(option)
                if plan.payments != expected:
                    failures.append("installment_schedule_mismatch")
                if plan.total_paid != option.total_payable_amount:
                    failures.append("installment_total_mismatch")

    elif plan.method is RecommendedPaymentMethod.WAIT:
        preference = considers(state, ConsideredPaymentMethod.FULL_PAYMENT)
        if earliest is None:
            failures.append("wait_requires_earliest_full_date")
        elif earliest <= request.request_date:
            failures.append("wait_requires_later_than_request_date")
        elif earliest > request.desired_completion_date:
            failures.append("wait_after_deadline")
        if len(plan.payments) != 1:
            failures.append("wait_requires_one_payment")
        else:
            if earliest is not None and plan.payments[0].date != earliest:
                failures.append("wait_date_must_equal_earliest")
            if plan.payments[0].amount != request.requested_amount:
                failures.append("wait_amount_must_equal_requested")

    else:
        failures.append(f"unknown_method_{plan.method.value}")

    if not preference:
        failures.append("preference_rejected")

    dates = [item.date for item in plan.payments]
    if dates != sorted(dates):
        failures.append("payments_not_chronological")
    previous: date | None = None
    for item in plan.payments:
        if item.amount <= _ZERO:
            failures.append("non_positive_payment")
        if previous is not None and item.date < previous:
            failures.append("payments_not_chronological")
        previous = item.date
        if item.date > horizon_end:
            failures.append("payment_outside_forecast_horizon")

    failures.extend(_spending_failures(state, plan.spending_changes))

    forecast = None
    financially_safe = False
    if plan.payments and "payment_outside_forecast_horizon" not in failures:
        forecast = simulate_plan(state, request, plan, config)
        financially_safe = forecast.is_safe
        if not financially_safe:
            failures.append("forecast_minimum_violated")
    elif not plan.payments:
        financially_safe = False
        failures.append("empty_payment_plan")

    completes = bool(plan.payments) and plan.completion_date is not None
    if completes and plan.completion_date is not None:
        completes = plan.completion_date <= request.desired_completion_date
    if plan.method is RecommendedPaymentMethod.WAIT and earliest is not None:
        completes = earliest <= request.desired_completion_date

    return CandidatePlan(
        method=plan.method,
        affordability_status=plan.affordability_status,
        payments=plan.payments,
        payment_option_id=plan.payment_option_id,
        spending_changes=plan.spending_changes,
        total_paid=plan.total_paid,
        start_date=plan.start_date,
        completion_date=plan.completion_date,
        completes_by_deadline=completes,
        requires_spending_changes=bool(plan.spending_changes),
        is_preference_eligible=preference,
        is_financially_safe=financially_safe,
        validation_failures=tuple(dict.fromkeys(failures)),
        forecast_result=forecast,
    )


def is_recommendable(plan: CandidatePlan) -> bool:
    return (
        plan.is_preference_eligible
        and plan.is_financially_safe
        and plan.completes_by_deadline
        and not plan.validation_failures
        and plan.method is not RecommendedPaymentMethod.NOT_RECOMMENDED
    )
