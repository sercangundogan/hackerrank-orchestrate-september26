"""Deterministic payment-candidate construction.

Capacity inputs (amount_safe_to_pay, earliest_date_for_full_payment) are
taken as given. This module does not search capacity or choose a winner.

Installment max-month rule (production):
    number_of_payments <= max_installment_months

A blank max_installment_months means the user will not consider installments.
The comparison is against the supplied option's payment count, not a homemade
duration. No request-specific exceptions.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from data.models import (
    AffordabilityStatus,
    ConsideredPaymentMethod,
    FinanceRequest,
    PaymentOption,
    PaymentOptionMethod,
    RecommendedPaymentMethod,
)
from decision.models import CandidatePlan, PlanPayment, SpendingAction
from finance.models import NormalizedFinancialState

_ZERO = Decimal("0")


def considers(state: NormalizedFinancialState, method: ConsideredPaymentMethod) -> bool:
    return method in state.profile.payment_methods_user_will_consider


def format_money(amount: Decimal) -> str:
    """Preserve integer amounts; pad non-integers to two decimal places."""
    if amount == amount.to_integral_value():
        return str(int(amount))
    return f"{amount:.2f}"


def format_payment_plan(payments: tuple[PlanPayment, ...]) -> str:
    if not payments:
        return "none"
    return "|".join(f"{item.date.isoformat()}:{format_money(item.amount)}" for item in payments)


def format_spending_changes(actions: tuple[SpendingAction, ...]) -> str:
    if not actions:
        return "none"
    parts: list[str] = []
    for action in actions:
        if action.kind.value == "stop":
            parts.append(f"stop:{action.event_id}")
        else:
            parts.append(f"reduce_to:{action.event_id}:{format_money(action.new_amount or _ZERO)}")
    return "|".join(parts)


def format_earliest(value: date | None) -> str:
    return "" if value is None else value.isoformat()


def installment_accepted(state: NormalizedFinancialState, option: PaymentOption) -> bool:
    """True when the option's payment count is within max_installment_months."""
    maximum = state.profile.max_installment_months
    if maximum is None:
        return False
    return option.number_of_payments <= maximum


def installment_schedule(option: PaymentOption) -> tuple[PlanPayment, ...]:
    """Exact seller schedule: first_payment_date + i * payment_frequency_days."""
    if option.payment_frequency_days is None:
        return ()
    return tuple(
        PlanPayment(
            date=option.first_payment_date
            + timedelta(days=i * option.payment_frequency_days),
            amount=option.payment_amount,
        )
        for i in range(option.number_of_payments)
    )


def _plan(
    *,
    method: RecommendedPaymentMethod,
    status: AffordabilityStatus,
    payments: tuple[PlanPayment, ...],
    option_id: str | None,
    spending: tuple[SpendingAction, ...],
    deadline: date,
    preference_eligible: bool,
) -> CandidatePlan:
    total = sum((item.amount for item in payments), _ZERO)
    start = payments[0].date if payments else None
    end = payments[-1].date if payments else None
    completes = bool(payments) and end is not None and end <= deadline
    return CandidatePlan(
        method=method,
        affordability_status=status,
        payments=payments,
        payment_option_id=option_id,
        spending_changes=spending,
        total_paid=total,
        start_date=start,
        completion_date=end,
        completes_by_deadline=completes,
        requires_spending_changes=bool(spending),
        is_preference_eligible=preference_eligible,
        is_financially_safe=False,
        validation_failures=(),
        forecast_result=None,
    )


def full_payment_candidate(
    request: FinanceRequest,
    spending: tuple[SpendingAction, ...] = (),
    *,
    preference_eligible: bool,
) -> CandidatePlan:
    payments = (PlanPayment(date=request.request_date, amount=request.requested_amount),)
    status = (
        AffordabilityStatus.AFFORDABLE_WITH_PLAN
        if spending
        else AffordabilityStatus.AFFORDABLE_NOW
    )
    return _plan(
        method=RecommendedPaymentMethod.FULL_PAYMENT,
        status=status,
        payments=payments,
        option_id=None,
        spending=spending,
        deadline=request.desired_completion_date,
        preference_eligible=preference_eligible,
    )


def partial_payment_candidate(
    request: FinanceRequest,
    amount_safe: Decimal,
    earliest: date,
    spending: tuple[SpendingAction, ...] = (),
    *,
    preference_eligible: bool,
) -> CandidatePlan:
    remainder = request.requested_amount - amount_safe
    payments = (
        PlanPayment(date=request.request_date, amount=amount_safe),
        PlanPayment(date=earliest, amount=remainder),
    )
    return _plan(
        method=RecommendedPaymentMethod.PARTIAL_PAYMENT,
        status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        payments=payments,
        option_id=None,
        spending=spending,
        deadline=request.desired_completion_date,
        preference_eligible=preference_eligible,
    )


def installment_candidate(
    request: FinanceRequest,
    option: PaymentOption,
    spending: tuple[SpendingAction, ...] = (),
    *,
    preference_eligible: bool,
) -> CandidatePlan:
    return _plan(
        method=RecommendedPaymentMethod.INSTALLMENTS,
        status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        payments=installment_schedule(option),
        option_id=option.payment_option_id,
        spending=spending,
        deadline=request.desired_completion_date,
        preference_eligible=preference_eligible,
    )


def wait_candidate(
    request: FinanceRequest,
    earliest: date,
    spending: tuple[SpendingAction, ...] = (),
    *,
    preference_eligible: bool,
) -> CandidatePlan:
    payments = (PlanPayment(date=earliest, amount=request.requested_amount),)
    return _plan(
        method=RecommendedPaymentMethod.WAIT,
        status=AffordabilityStatus.AFFORDABLE_LATER,
        payments=payments,
        option_id=None,
        spending=spending,
        deadline=request.desired_completion_date,
        preference_eligible=preference_eligible,
    )


def not_recommended_candidate(request: FinanceRequest) -> CandidatePlan:
    return CandidatePlan(
        method=RecommendedPaymentMethod.NOT_RECOMMENDED,
        affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
        payments=(),
        payment_option_id=None,
        spending_changes=(),
        total_paid=_ZERO,
        start_date=None,
        completion_date=None,
        completes_by_deadline=False,
        requires_spending_changes=False,
        is_preference_eligible=True,
        is_financially_safe=True,
        validation_failures=(),
        forecast_result=None,
    )


def enumerate_baseline_candidates(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    options: tuple[PaymentOption, ...],
    amount_safe: Decimal,
    earliest: date | None,
    full_safe_today: bool,
    spending: tuple[SpendingAction, ...] = (),
) -> tuple[CandidatePlan, ...]:
    """Build preference-eligible baseline (or spending-assisted) candidates.

    Full payment without spending changes is emitted only when the Phase 5A
    capacity result says the requested amount is safe today. Spending-assisted
    full payment is still emitted so the simulator can confirm it.
    """
    plans: list[CandidatePlan] = []
    prefers_full = considers(state, ConsideredPaymentMethod.FULL_PAYMENT)
    prefers_partial = considers(state, ConsideredPaymentMethod.PARTIAL_PAYMENT)
    prefers_installments = considers(state, ConsideredPaymentMethod.INSTALLMENTS)

    if prefers_full and (full_safe_today or spending):
        plans.append(
            full_payment_candidate(
                request, spending, preference_eligible=True
            )
        )

    if (
        prefers_partial
        and request.allows_partial_payment
        and _ZERO < amount_safe < request.requested_amount
        and earliest is not None
        and earliest <= request.desired_completion_date
    ):
        plans.append(
            partial_payment_candidate(
                request,
                amount_safe,
                earliest,
                spending,
                preference_eligible=True,
            )
        )

    if prefers_installments and state.profile.max_installment_months is not None:
        for option in options:
            if option.payment_method is not PaymentOptionMethod.INSTALLMENTS:
                continue
            plans.append(
                installment_candidate(
                    request, option, spending, preference_eligible=True
                )
            )

    if (
        prefers_full
        and earliest is not None
        and earliest > request.request_date
        and earliest <= request.desired_completion_date
    ):
        plans.append(
            wait_candidate(request, earliest, spending, preference_eligible=True)
        )

    return tuple(plans)
