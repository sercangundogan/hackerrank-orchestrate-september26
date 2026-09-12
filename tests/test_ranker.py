from __future__ import annotations

from datetime import date
from decimal import Decimal

from data.models import AffordabilityStatus, RecommendedPaymentMethod
from decision.models import CandidatePlan, SpendingAction, SpendingActionKind
from decision.ranker import official_rank_key, rank_candidates, rank_key
from decision.models import PlanPayment


def _plan(**overrides: object) -> CandidatePlan:
    values: dict[str, object] = {
        "method": RecommendedPaymentMethod.FULL_PAYMENT,
        "affordability_status": AffordabilityStatus.AFFORDABLE_NOW,
        "payments": (PlanPayment(date=date(2026, 3, 3), amount=Decimal("100")),),
        "payment_option_id": None,
        "spending_changes": (),
        "total_paid": Decimal("100"),
        "start_date": date(2026, 3, 3),
        "completion_date": date(2026, 3, 3),
        "completes_by_deadline": True,
        "requires_spending_changes": False,
        "is_preference_eligible": True,
        "is_financially_safe": True,
        "validation_failures": (),
        "forecast_result": None,
    }
    values.update(overrides)
    return CandidatePlan(**values)  # type: ignore[arg-type]


def _change(event_id: str, amount: Decimal = Decimal("10")) -> SpendingAction:
    return SpendingAction(
        kind=SpendingActionKind.STOP,
        event_id=event_id,
        new_amount=None,
        series_event_ids=(event_id,),
        category="streaming",
        normal_amount=amount,
        series_key=event_id,
    )


def test_deadline_beats_everything() -> None:
    winner = _plan(completes_by_deadline=True, total_paid=Decimal("500"))
    loser = _plan(
        completes_by_deadline=False,
        total_paid=Decimal("10"),
        requires_spending_changes=False,
        start_date=date(2026, 3, 1),
    )
    assert rank_candidates((loser, winner))[0] is winner


def test_no_spending_changes_beats_changes() -> None:
    winner = _plan(requires_spending_changes=False, total_paid=Decimal("200"))
    loser = _plan(
        requires_spending_changes=True,
        spending_changes=(_change("event_a"),),
        total_paid=Decimal("100"),
    )
    assert rank_candidates((loser, winner))[0] is winner


def test_minimize_total_paid() -> None:
    winner = _plan(total_paid=Decimal("100"), payment_option_id="payment_option_09")
    loser = _plan(total_paid=Decimal("150"), payment_option_id="payment_option_01")
    assert rank_candidates((loser, winner))[0] is winner


def test_earlier_start_wins() -> None:
    winner = _plan(start_date=date(2026, 3, 3), total_paid=Decimal("100"))
    loser = _plan(start_date=date(2026, 3, 10), total_paid=Decimal("100"))
    assert rank_candidates((loser, winner))[0] is winner


def test_fewer_payments_wins() -> None:
    winner = _plan(
        payments=(PlanPayment(date=date(2026, 3, 3), amount=Decimal("100")),),
        total_paid=Decimal("100"),
        start_date=date(2026, 3, 3),
    )
    loser = _plan(
        payments=(
            PlanPayment(date=date(2026, 3, 3), amount=Decimal("50")),
            PlanPayment(date=date(2026, 3, 10), amount=Decimal("50")),
        ),
        total_paid=Decimal("100"),
        start_date=date(2026, 3, 3),
    )
    assert rank_candidates((loser, winner))[0] is winner


def test_lowest_option_id_wins() -> None:
    winner = _plan(payment_option_id="payment_option_01", total_paid=Decimal("100"))
    loser = _plan(payment_option_id="payment_option_02", total_paid=Decimal("100"))
    none = _plan(payment_option_id=None, total_paid=Decimal("100"))
    ranked = rank_candidates((none, loser, winner))
    assert ranked[0] is winner
    assert official_rank_key(winner) < official_rank_key(loser)
    assert official_rank_key(winner) < official_rank_key(none)


def test_spending_tiebreak_after_official_keys() -> None:
    fewer = _plan(
        requires_spending_changes=True,
        spending_changes=(_change("event_b", Decimal("5")),),
        total_paid=Decimal("100"),
    )
    more = _plan(
        requires_spending_changes=True,
        spending_changes=(_change("event_a", Decimal("1")), _change("event_c", Decimal("1"))),
        total_paid=Decimal("100"),
    )
    assert official_rank_key(fewer) == official_rank_key(more)
    assert rank_key(fewer) < rank_key(more)
    assert rank_candidates((more, fewer))[0] is fewer
