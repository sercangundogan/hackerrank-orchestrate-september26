"""Deterministic decision explanations. No LLM."""

from __future__ import annotations

from data.models import FinanceRequest, RecommendedPaymentMethod
from decision.models import CapacityResult, CandidatePlan, ExplanationFacts, SpendingActionKind
from decision.plans import format_money
from finance.forecast_models import ForecastEventKind
from finance.models import NormalizedFinancialState


def _upcoming_obligations(capacity: CapacityResult) -> tuple[str, ...]:
    summaries: list[str] = []
    if capacity.limiting_event is not None and capacity.limiting_date is not None:
        entry = capacity.limiting_event
        label = entry.category or entry.kind.value
        summaries.append(
            f"{label} on {capacity.limiting_date.isoformat()} "
            f"({format_money(entry.amount_home_currency)})"
        )
    kinds = {
        ForecastEventKind.PENDING_DEBIT_RESERVE,
        ForecastEventKind.SCHEDULED_DEBIT,
        ForecastEventKind.GENERATED_RECURRING_DEBIT,
    }
    seen: set[str] = set()
    for entry in capacity.baseline.all_entries:
        if entry.kind not in kinds or entry.signed_amount >= 0:
            continue
        key = f"{entry.date.isoformat()}:{entry.category}"
        if key in seen:
            continue
        seen.add(key)
        summaries.append(f"{entry.category} on {entry.date.isoformat()}")
        if len(summaries) >= 4:
            break
    return tuple(summaries)


def _income_date(capacity: CapacityResult):
    for entry in capacity.baseline.all_entries:
        if entry.signed_amount > 0 and entry.category == "salary":
            return entry.date
    return None


def _spending_summaries(plan: CandidatePlan) -> tuple[str, ...]:
    lines: list[str] = []
    for action in plan.spending_changes:
        if action.kind is SpendingActionKind.STOP:
            lines.append(f"stop {action.category} ({action.event_id})")
        else:
            amount = format_money(action.new_amount) if action.new_amount is not None else "0"
            lines.append(f"reduce {action.category} to {amount} ({action.event_id})")
    return tuple(lines)


def _text(
    request: FinanceRequest,
    state: NormalizedFinancialState,
    capacity: CapacityResult,
    plan: CandidatePlan,
    currency: str,
    min_keep: str,
) -> str:
    requested = format_money(request.requested_amount)
    safe = format_money(capacity.amount_safe_to_pay)
    deadline = request.desired_completion_date.isoformat()
    method = plan.method
    if method is RecommendedPaymentMethod.NOT_RECOMMENDED:
        if capacity.amount_safe_to_pay > 0:
            return (
                f"Do not proceed with the {currency} {requested} request. Although "
                f"{currency} {safe} is available today, the full amount cannot be "
                f"completed safely by {deadline}."
            )
        return (
            f"Do not make this payment by {deadline}. None of the available "
            f"options keeps the {currency} {min_keep} minimum protected."
        )
    if method is RecommendedPaymentMethod.FULL_PAYMENT:
        if plan.spending_changes:
            changes = "; ".join(_spending_summaries(plan))
            return (
                f"{changes.capitalize()}, then pay {currency} {requested} today. "
                f"This leaves at least {currency} {min_keep} available."
            )
        return (
            f"Pay {currency} {requested} today. This leaves at least "
            f"{currency} {min_keep} available over the next 90 days."
        )
    if method is RecommendedPaymentMethod.WAIT and plan.payments:
        when = plan.payments[0].date.isoformat()
        income = _income_date(capacity)
        if income is not None and income == plan.payments[0].date:
            return (
                f"Wait until confirmed salary on {when}, then pay {currency} "
                f"{requested} in full. Paying earlier would take the balance "
                f"below the {currency} {min_keep} minimum."
            )
        return (
            f"Pay {currency} {requested} in full on {when}. Paying earlier would "
            f"take the balance below the {currency} {min_keep} minimum."
        )
    if method is RecommendedPaymentMethod.PARTIAL_PAYMENT and len(plan.payments) == 2:
        first, second = plan.payments
        return (
            f"Pay {currency} {format_money(first.amount)} today and the remaining "
            f"{currency} {format_money(second.amount)} on {second.date.isoformat()}. "
            f"This completes the full request and keeps the {currency} {min_keep} "
            f"minimum protected."
        )
    if method is RecommendedPaymentMethod.INSTALLMENTS and plan.payments:
        each = format_money(plan.payments[0].amount)
        start = plan.payments[0].date.isoformat()
        return (
            f"Use {len(plan.payments)} installments of {currency} {each}, starting "
            f"{start}. This leaves at least {currency} {min_keep} available."
        )
    return (
        f"Recommend {method.value} for {currency} {requested} with status "
        f"{plan.affordability_status.value}."
    )


def build_explanation(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    capacity: CapacityResult,
    plan: CandidatePlan,
) -> ExplanationFacts:
    currency = state.profile.home_currency.value
    min_keep = format_money(state.profile.minimum_balance_to_keep)
    fee = None
    total = plan.total_paid if plan.method is RecommendedPaymentMethod.INSTALLMENTS else None
    text = _text(request, state, capacity, plan, currency, min_keep)
    return ExplanationFacts(
        safe_amount_today=capacity.amount_safe_to_pay,
        requested_amount=request.requested_amount,
        minimum_balance_to_keep=state.profile.minimum_balance_to_keep,
        home_currency=currency,
        upcoming_obligation_summaries=_upcoming_obligations(capacity),
        confirmed_income_date=_income_date(capacity),
        installment_total=total,
        installment_fee=fee,
        spending_change_summaries=_spending_summaries(plan),
        chosen_method=plan.method.value,
        chosen_status=plan.affordability_status.value,
        completes_by_deadline=plan.completes_by_deadline,
        baseline_full_safe_today=capacity.full_payment_safe_today,
        text=text,
    )
