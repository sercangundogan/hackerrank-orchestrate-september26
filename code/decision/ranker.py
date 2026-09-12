"""Official payment-plan ranking.

Keys, in order:

1. Completes the request by desired_completion_date
2. Requires no spending changes
3. Minimize total amount paid
4. Start payment earlier
5. Use fewer payments
6. Lowest payment_option_id

A sentinel option id is used only for key 6 when a plan has no seller option.
After those official keys are equal, an internal spending-change tie-break
applies: fewer changes, then smaller total reduction from normal spending,
then event-id order. That extra key is never placed ahead of the official
rules.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from decision.models import RANKING_OPTION_ID_SENTINEL, CandidatePlan, SpendingActionKind

_ZERO = Decimal("0")
_FAR_DATE = date(9999, 12, 31)


def official_rank_key(plan: CandidatePlan) -> tuple:
    start = plan.start_date if plan.start_date is not None else _FAR_DATE
    option_id = plan.payment_option_id or RANKING_OPTION_ID_SENTINEL
    return (
        0 if plan.completes_by_deadline else 1,
        1 if plan.requires_spending_changes else 0,
        plan.total_paid,
        start,
        len(plan.payments),
        option_id,
    )


def spending_tiebreak_key(plan: CandidatePlan) -> tuple:
    reduction = _ZERO
    for action in plan.spending_changes:
        if action.kind is SpendingActionKind.STOP:
            reduction += action.normal_amount
        elif action.new_amount is not None:
            reduction += action.normal_amount - action.new_amount
    event_ids = tuple(sorted(action.event_id for action in plan.spending_changes))
    return (len(plan.spending_changes), reduction, event_ids)


def rank_key(plan: CandidatePlan) -> tuple:
    return official_rank_key(plan) + spending_tiebreak_key(plan)


def rank_candidates(plans: tuple[CandidatePlan, ...]) -> tuple[CandidatePlan, ...]:
    return tuple(sorted(plans, key=rank_key))
