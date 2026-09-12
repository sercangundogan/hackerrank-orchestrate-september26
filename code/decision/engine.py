"""Assemble, validate, and rank payment candidates into a DecisionResult.

amount_safe_to_pay and earliest_date_for_full_payment come from Phase 5A
and are copied unchanged, including when spending changes unlock a plan.
"""

from __future__ import annotations

from data.models import FinanceRequest, PaymentOption
from decision.capacity import compute_capacity
from decision.explanation import build_explanation
from decision.models import CapacityResult, CandidatePlan, DecisionResult
from decision.plans import enumerate_baseline_candidates, not_recommended_candidate
from decision.ranker import rank_candidates
from decision.spending_optimizer import action_combinations, legal_actions
from decision.validator import is_recommendable, validate_candidate
from finance.forecast_models import ForecastConfig
from finance.models import NormalizedFinancialState


def _validate_all(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    options: tuple[PaymentOption, ...],
    plans: tuple[CandidatePlan, ...],
    capacity: CapacityResult,
    config: ForecastConfig,
) -> list[CandidatePlan]:
    return [
        validate_candidate(
            state,
            request,
            plan,
            options,
            capacity.amount_safe_to_pay,
            capacity.earliest_date_for_full_payment,
            config,
        )
        for plan in plans
    ]


def decide(
    state: NormalizedFinancialState,
    request: FinanceRequest,
    payment_options: tuple[PaymentOption, ...],
    config: ForecastConfig | None = None,
    *,
    capacity: CapacityResult | None = None,
) -> DecisionResult:
    strategy = config or ForecastConfig(strict_unresolved_amounts=True)
    capacity_result = capacity or compute_capacity(state, request, strategy)
    validated: list[CandidatePlan] = []

    baseline = enumerate_baseline_candidates(
        state,
        request,
        payment_options,
        capacity_result.amount_safe_to_pay,
        capacity_result.earliest_date_for_full_payment,
        capacity_result.full_payment_safe_today,
    )
    validated.extend(
        _validate_all(state, request, payment_options, baseline, capacity_result, strategy)
    )

    has_safe = any(is_recommendable(plan) for plan in validated)
    if not has_safe:
        actions = legal_actions(state, capacity_result.baseline)
        for combo in action_combinations(actions):
            assisted = enumerate_baseline_candidates(
                state,
                request,
                payment_options,
                capacity_result.amount_safe_to_pay,
                capacity_result.earliest_date_for_full_payment,
                capacity_result.full_payment_safe_today,
                combo,
            )
            validated.extend(
                _validate_all(
                    state, request, payment_options, assisted, capacity_result, strategy
                )
            )

    recommendable = tuple(plan for plan in validated if is_recommendable(plan))
    if recommendable:
        chosen = rank_candidates(recommendable)[0]
    else:
        chosen = validate_candidate(
            state,
            request,
            not_recommended_candidate(request),
            payment_options,
            capacity_result.amount_safe_to_pay,
            capacity_result.earliest_date_for_full_payment,
            strategy,
        )

    rejected = tuple(plan for plan in validated if plan != chosen)
    explanation = build_explanation(state, request, capacity_result, chosen)
    return DecisionResult(
        request_id=request.request_id,
        amount_safe_to_pay=capacity_result.amount_safe_to_pay,
        affordability_status=chosen.affordability_status,
        recommended_payment_method=chosen.method,
        payment_plan=chosen.payments,
        earliest_date_for_full_payment=capacity_result.earliest_date_for_full_payment,
        spending_changes_needed=chosen.spending_changes,
        explanation_facts=explanation,
        chosen_candidate=chosen,
        rejected_candidates=rejected,
    )
