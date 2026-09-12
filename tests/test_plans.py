from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from data.models import (
    AffordabilityStatus,
    ConsideredPaymentMethod,
    EventDirection,
    EventStatus,
    EventType,
    Flexibility,
    PaymentOption,
    PaymentOptionMethod,
    RecommendedPaymentMethod,
)
from decision.capacity import compute_capacity
from decision.engine import decide
from decision.models import CandidatePlan, SpendingAction, SpendingActionKind
from decision.plans import (
    format_payment_plan,
    format_spending_changes,
    full_payment_candidate,
    installment_schedule,
    not_recommended_candidate,
    partial_payment_candidate,
    wait_candidate,
)
from decision.spending_optimizer import (
    action_combinations,
    latest_series_event_id,
    legal_actions,
)
from decision.validator import is_recommendable, validate_candidate
from finance.forecast import forecast_financial_state
from finance.forecast_models import ForecastConfig, RecurrenceOverride
from finance.models import Cadence, CadenceConfidence
from tests.test_capacity import _request
from tests.test_forecast import _event, _profile, _series, _state

CONFIG = ForecastConfig(essential_spend_enabled=False, strict_unresolved_amounts=True)


def _option(**overrides: object) -> PaymentOption:
    values: dict[str, object] = {
        "payment_option_id": "payment_option_test_1",
        "request_id": "request_test",
        "payment_method": PaymentOptionMethod.INSTALLMENTS,
        "payment_amount": Decimal("400"),
        "number_of_payments": 3,
        "first_payment_date": date(2026, 3, 3),
        "payment_frequency_days": 30,
        "financing_fee": Decimal("200"),
        "total_payable_amount": Decimal("1200"),
    }
    values.update(overrides)
    return PaymentOption(**values)  # type: ignore[arg-type]


def _streaming_series(**overrides: object):
    values: dict[str, object] = {
        "category": "streaming",
        "normalized_description": "family streaming",
        "original_description": "Family streaming",
        "event_ids": ("event_s1", "event_s2", "event_s3", "event_s4"),
        "event_type": EventType.SUBSCRIPTION,
        "observed_dates": (
            date(2026, 2, 3),
            date(2026, 2, 10),
            date(2026, 2, 17),
            date(2026, 2, 24),
        ),
        "observed_amounts_home_currency": (
            Decimal("200"),
            Decimal("200"),
            Decimal("200"),
            Decimal("200"),
        ),
        "inferred_cadence": Cadence.WEEKLY,
        "inferred_cadence_days": 7,
        "cadence_confidence": CadenceConfidence.HIGH,
        "representative_amount": Decimal("200"),
        "flexibility": Flexibility.STOPPABLE,
        "minimum_allowed_amount": None,
        "is_protected_category": False,
        "profile_allows_reduce": False,
        "profile_allows_stop": True,
        "likely_recurring": True,
    }
    values.update(overrides)
    return _series(**values)


def _validate(state, request, plan, options=(), amount_safe=None, earliest=None):
    capacity = compute_capacity(state, request, CONFIG)
    return validate_candidate(
        state,
        request,
        plan,
        options,
        amount_safe if amount_safe is not None else capacity.amount_safe_to_pay,
        earliest if earliest is not None else capacity.earliest_date_for_full_payment,
        CONFIG,
    )


def test_full_safe_today() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("2000"))
    result = decide(state, request, (), CONFIG)
    assert result.recommended_payment_method is RecommendedPaymentMethod.FULL_PAYMENT
    assert result.affordability_status is AffordabilityStatus.AFFORDABLE_NOW
    assert result.payment_plan[0].date == request.request_date
    assert result.payment_plan[0].amount == Decimal("2000")
    assert result.spending_changes_needed == ()
    assert result.earliest_date_for_full_payment == request.request_date


def test_full_preference_rejection() -> None:
    state = _state(
        profile=_profile(
            payment_methods_user_will_consider=(ConsideredPaymentMethod.INSTALLMENTS,),
            max_installment_months=6,
        )
    )
    request = _request(requested_amount=Decimal("2000"))
    option = _option(
        payment_amount=Decimal("700"),
        number_of_payments=3,
        total_payable_amount=Decimal("2100"),
        financing_fee=Decimal("100"),
        first_payment_date=date(2026, 3, 4),
    )
    plan = full_payment_candidate(request, preference_eligible=False)
    checked = _validate(state, request, plan, (option,))
    assert checked.is_preference_eligible is False
    assert "preference_rejected" in checked.validation_failures
    result = decide(state, request, (option,), CONFIG)
    assert result.recommended_payment_method is not RecommendedPaymentMethod.FULL_PAYMENT


def test_partial_request_disallows() -> None:
    state = _state(
        profile=_profile(
            payment_methods_user_will_consider=(
                ConsideredPaymentMethod.FULL_PAYMENT,
                ConsideredPaymentMethod.PARTIAL_PAYMENT,
            )
        )
    )
    request = _request(requested_amount=Decimal("8000"), allows_partial_payment=False)
    capacity = compute_capacity(state, request, CONFIG)
    plan = partial_payment_candidate(
        request,
        capacity.amount_safe_to_pay,
        capacity.earliest_date_for_full_payment or request.request_date,
        preference_eligible=True,
    )
    checked = _validate(state, request, plan)
    assert "request_disallows_partial" in checked.validation_failures
    result = decide(state, request, (), CONFIG)
    assert result.recommended_payment_method is not RecommendedPaymentMethod.PARTIAL_PAYMENT


def test_partial_preference_rejection() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("8000"))
    capacity = compute_capacity(state, request, CONFIG)
    plan = partial_payment_candidate(
        request,
        capacity.amount_safe_to_pay,
        date(2026, 3, 10),
        preference_eligible=False,
    )
    checked = _validate(state, request, plan, amount_safe=capacity.amount_safe_to_pay, earliest=date(2026, 3, 10))
    assert checked.is_preference_eligible is False


def test_partial_zero_and_full_safe_rejected() -> None:
    rich = _state(profile=_profile(current_available_balance=Decimal("20000")))
    request = _request(requested_amount=Decimal("150"))
    capacity = compute_capacity(rich, request, CONFIG)
    assert capacity.amount_safe_to_pay == request.requested_amount
    plan = partial_payment_candidate(
        request, Decimal("50"), date(2026, 3, 10), preference_eligible=True
    )
    checked = _validate(rich, request, plan, amount_safe=capacity.amount_safe_to_pay, earliest=date(2026, 3, 10))
    assert "partial_requires_strict_partial_safe_amount" in checked.validation_failures

    empty = _state(profile=_profile(current_available_balance=Decimal("1000")))
    big = _request(requested_amount=Decimal("400"))
    zero_cap = compute_capacity(empty, big, CONFIG)
    assert zero_cap.amount_safe_to_pay == Decimal("0")
    zero_plan = partial_payment_candidate(
        big, Decimal("0"), date(2026, 3, 10), preference_eligible=True
    )
    zero_checked = _validate(
        empty, big, zero_plan, amount_safe=zero_cap.amount_safe_to_pay, earliest=date(2026, 3, 10)
    )
    assert "partial_requires_strict_partial_safe_amount" in zero_checked.validation_failures


def test_partial_exactly_two_payments_and_deadline() -> None:
    state = _state(
        profile=_profile(
            payment_methods_user_will_consider=(
                ConsideredPaymentMethod.FULL_PAYMENT,
                ConsideredPaymentMethod.PARTIAL_PAYMENT,
            )
        )
    )
    request = _request(requested_amount=Decimal("8000"), desired_completion_date=date(2026, 3, 5))
    capacity = compute_capacity(state, request, CONFIG)
    assert Decimal("0") < capacity.amount_safe_to_pay < request.requested_amount
    late = request.desired_completion_date + timedelta(days=10)
    plan = partial_payment_candidate(
        request, capacity.amount_safe_to_pay, late, preference_eligible=True
    )
    checked = _validate(state, request, plan, earliest=late)
    assert "partial_second_payment_after_deadline" in checked.validation_failures
    assert len(plan.payments) == 2
    assert plan.payments[0].amount + plan.payments[1].amount == request.requested_amount


def test_partial_combined_schedule_safety() -> None:
    from decision.capacity import candidate_schedule
    from decision.models import PlanPayment

    state = _state()
    request = _request(requested_amount=Decimal("5000"))
    plan = CandidatePlan(
        method=RecommendedPaymentMethod.PARTIAL_PAYMENT,
        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        payments=(
            PlanPayment(date=date(2026, 3, 3), amount=Decimal("2500")),
            PlanPayment(date=date(2026, 3, 13), amount=Decimal("2500")),
        ),
        payment_option_id=None,
        spending_changes=(),
        total_paid=Decimal("5000"),
        start_date=date(2026, 3, 3),
        completion_date=date(2026, 3, 13),
        completes_by_deadline=True,
        requires_spending_changes=False,
        is_preference_eligible=True,
        is_financially_safe=False,
        validation_failures=(),
    )
    alone_today = forecast_financial_state(
        state,
        CONFIG,
        candidate_payments=candidate_schedule(request, ((date(2026, 3, 3), Decimal("2500")),)),
    )
    alone_later = forecast_financial_state(
        state,
        CONFIG,
        candidate_payments=candidate_schedule(request, ((date(2026, 3, 13), Decimal("2500")),)),
    )
    assert alone_today.is_safe
    assert alone_later.is_safe
    checked = _validate(
        state, request, plan, amount_safe=Decimal("2500"), earliest=date(2026, 3, 13)
    )
    assert checked.is_financially_safe is False
    assert "forecast_minimum_violated" in checked.validation_failures


def test_installment_exact_schedule_and_fee() -> None:
    option = _option()
    schedule = installment_schedule(option)
    assert len(schedule) == 3
    assert schedule[0].date == date(2026, 3, 3)
    assert schedule[1].date == date(2026, 4, 2)
    assert schedule[2].date == date(2026, 5, 2)
    assert all(item.amount == Decimal("400") for item in schedule)
    assert sum((item.amount for item in schedule), Decimal("0")) == option.total_payable_amount


def test_installment_max_constraint_and_deadline_and_preference() -> None:
    state = _state(
        profile=_profile(
            payment_methods_user_will_consider=(ConsideredPaymentMethod.INSTALLMENTS,),
            max_installment_months=2,
        )
    )
    request = _request(requested_amount=Decimal("1000"), desired_completion_date=date(2026, 3, 20))
    too_many = _option(number_of_payments=3, payment_amount=Decimal("400"), total_payable_amount=Decimal("1200"))
    late = _option(
        payment_option_id="payment_option_late",
        number_of_payments=2,
        payment_amount=Decimal("550"),
        total_payable_amount=Decimal("1100"),
        first_payment_date=date(2026, 3, 15),
        payment_frequency_days=30,
    )
    from decision.plans import installment_candidate

    many = _validate(state, request, installment_candidate(request, too_many, preference_eligible=True), (too_many,))
    assert "exceeds_max_installment_months" in many.validation_failures
    late_plan = _validate(state, request, installment_candidate(request, late, preference_eligible=True), (late,))
    assert late_plan.completes_by_deadline is False

    no_pref = _state(profile=_profile(max_installment_months=6))
    pref = _validate(
        no_pref,
        request,
        installment_candidate(request, too_many, preference_eligible=True),
        (too_many,),
    )
    assert pref.is_preference_eligible is False


def test_installment_unsafe_schedule() -> None:
    state = _state(profile=_profile(
        current_available_balance=Decimal("1500"),
        payment_methods_user_will_consider=(ConsideredPaymentMethod.INSTALLMENTS,),
        max_installment_months=6,
    ))
    request = _request(requested_amount=Decimal("1000"))
    option = _option(
        payment_amount=Decimal("800"),
        number_of_payments=2,
        total_payable_amount=Decimal("1600"),
        financing_fee=Decimal("600"),
        first_payment_date=date(2026, 3, 3),
        payment_frequency_days=7,
    )
    from decision.plans import installment_candidate

    checked = _validate(
        state, request, installment_candidate(request, option, preference_eligible=True), (option,)
    )
    assert checked.is_financially_safe is False


def test_wait_later_safe_and_after_deadline_and_preference() -> None:
    salary = _event(
        source_event_id="sal_1",
        cash_date=date(2026, 3, 20),
        event_date=date(2026, 3, 20),
        settlement_date=date(2026, 3, 20),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("8000"),
        original_amount=Decimal("8000"),
        event_type=EventType.INCOME,
        category="salary",
        description="Monthly salary",
        status=EventStatus.SCHEDULED,
        is_historical=False,
        is_prospective=True,
    )
    state = _state(
        confirmed_scheduled_income=(salary,),
        classified_events=(salary,),
        profile=_profile(current_available_balance=Decimal("2000")),
    )
    request = _request(requested_amount=Decimal("2500"), desired_completion_date=date(2026, 3, 25))
    result = decide(state, request, (), CONFIG)
    assert result.recommended_payment_method is RecommendedPaymentMethod.WAIT
    assert result.affordability_status is AffordabilityStatus.AFFORDABLE_LATER

    late_request = _request(requested_amount=Decimal("2500"), desired_completion_date=date(2026, 3, 10))
    late_capacity = compute_capacity(state, late_request, CONFIG)
    assert late_capacity.earliest_date_for_full_payment is not None
    late_plan = wait_candidate(
        late_request, late_capacity.earliest_date_for_full_payment, preference_eligible=True
    )
    late_checked = _validate(state, late_request, late_plan)
    assert "wait_after_deadline" in late_checked.validation_failures or not is_recommendable(late_checked)

    no_full = _state(
        confirmed_scheduled_income=(salary,),
        classified_events=(salary,),
        profile=_profile(
            current_available_balance=Decimal("2000"),
            payment_methods_user_will_consider=(ConsideredPaymentMethod.INSTALLMENTS,),
            max_installment_months=6,
        ),
    )
    wait_pref = wait_candidate(request, date(2026, 3, 20), preference_eligible=True)
    wait_checked = _validate(no_full, request, wait_pref, earliest=date(2026, 3, 20))
    assert wait_checked.is_preference_eligible is False


def test_spending_stop_and_reduce_legal() -> None:
    series = _streaming_series()
    dining = _series(
        category="dining",
        normalized_description="weekend dining",
        original_description="Weekend dining",
        event_ids=("event_d1", "event_d2", "event_d3", "event_d4"),
        event_type=EventType.EXPENSE,
        observed_dates=(date(2026, 2, 3), date(2026, 2, 10), date(2026, 2, 17), date(2026, 2, 24)),
        observed_amounts_home_currency=(Decimal("200"), Decimal("200"), Decimal("200"), Decimal("200")),
        inferred_cadence=Cadence.WEEKLY,
        inferred_cadence_days=7,
        cadence_confidence=CadenceConfidence.HIGH,
        representative_amount=Decimal("200"),
        flexibility=Flexibility.REDUCIBLE,
        minimum_allowed_amount=Decimal("50"),
        is_protected_category=False,
        profile_allows_reduce=True,
        profile_allows_stop=False,
        likely_recurring=True,
    )
    state = _state(
        profile=_profile(current_available_balance=Decimal("2500")),
        recurring_series_candidates=(series, dining),
    )
    forecast = forecast_financial_state(state, CONFIG)
    actions = legal_actions(state, forecast)
    kinds = {(item.kind, item.event_id) for item in actions}
    assert (SpendingActionKind.STOP, latest_series_event_id(series)) in kinds
    assert (SpendingActionKind.REDUCE_TO, latest_series_event_id(dining)) in kinds


def test_spending_protected_and_fixed_rejected() -> None:
    request = _request(requested_amount=Decimal("100"))
    rent = _series(
        category="rent",
        event_ids=("event_rent",),
        flexibility=Flexibility.FIXED,
        is_protected_category=True,
        representative_amount=Decimal("1000"),
        observed_dates=(date(2026, 2, 3),),
        observed_amounts_home_currency=(Decimal("1000"),),
    )
    state = _state(recurring_series_candidates=(rent,))
    action = SpendingAction(
        kind=SpendingActionKind.STOP,
        event_id="event_rent",
        new_amount=None,
        series_event_ids=("event_rent",),
        category="rent",
        normal_amount=Decimal("1000"),
        series_key="rent",
    )
    plan = full_payment_candidate(request, (action,), preference_eligible=True)
    checked = _validate(state, request, plan)
    assert any("protected" in item or "fixed" in item for item in checked.validation_failures)


def test_spending_more_than_three_and_duplicate_rejected() -> None:
    request = _request()
    series_ids = [f"event_x{i}" for i in range(4)]
    extras = []
    for event_id in series_ids:
        extras.append(
            _streaming_series(
                event_ids=(event_id,),
                category="streaming",
                normalized_description=event_id,
                original_description=event_id,
                observed_dates=(date(2026, 2, 24),),
                observed_amounts_home_currency=(Decimal("10"),),
                representative_amount=Decimal("10"),
            )
        )
    state = _state(recurring_series_candidates=tuple(extras))
    actions = tuple(
        SpendingAction(
            kind=SpendingActionKind.STOP,
            event_id=event_id,
            new_amount=None,
            series_event_ids=(event_id,),
            category="streaming",
            normal_amount=Decimal("10"),
            series_key=event_id,
        )
        for event_id in series_ids
    )
    plan = full_payment_candidate(request, actions, preference_eligible=True)
    checked = _validate(state, request, plan)
    assert "more_than_three_spending_changes" in checked.validation_failures

    dup = (
        SpendingAction(
            kind=SpendingActionKind.STOP,
            event_id="event_s4",
            new_amount=None,
            series_event_ids=("event_s4",),
            category="streaming",
            normal_amount=Decimal("200"),
            series_key="s",
        ),
        SpendingAction(
            kind=SpendingActionKind.REDUCE_TO,
            event_id="event_s4",
            new_amount=Decimal("1"),
            series_event_ids=("event_s4",),
            category="streaming",
            normal_amount=Decimal("200"),
            series_key="s",
        ),
    )
    stream = _streaming_series(flexibility=Flexibility.REDUCIBLE_OR_STOPPABLE, minimum_allowed_amount=Decimal("1"), profile_allows_reduce=True)
    dup_state = _state(recurring_series_candidates=(stream,))
    dup_plan = full_payment_candidate(request, dup, preference_eligible=True)
    dup_checked = _validate(dup_state, request, dup_plan)
    assert any("duplicate" in item for item in dup_checked.validation_failures)


def test_counterfactual_full_becomes_safe_and_amount_unchanged() -> None:
    series = _streaming_series()
    state = _state(
        profile=_profile(current_available_balance=Decimal("2500")),
        recurring_series_candidates=(series,),
    )
    request = _request(requested_amount=Decimal("1400"), desired_completion_date=date(2026, 3, 10))
    capacity = compute_capacity(state, request, CONFIG)
    assert capacity.full_payment_safe_today is False
    result = decide(state, request, (), CONFIG, capacity=capacity)
    assert result.amount_safe_to_pay == capacity.amount_safe_to_pay
    assert result.earliest_date_for_full_payment == capacity.earliest_date_for_full_payment
    assert result.recommended_payment_method is RecommendedPaymentMethod.FULL_PAYMENT
    assert result.affordability_status is AffordabilityStatus.AFFORDABLE_WITH_PLAN
    assert result.spending_changes_needed
    assert result.spending_changes_needed[0].kind is SpendingActionKind.STOP
    assert result.spending_changes_needed[0].event_id == latest_series_event_id(series)


def test_action_combinations_skip_same_series() -> None:
    stop = SpendingAction(
        kind=SpendingActionKind.STOP,
        event_id="event_a",
        new_amount=None,
        series_event_ids=("event_a",),
        category="streaming",
        normal_amount=Decimal("10"),
        series_key="same",
    )
    reduce = SpendingAction(
        kind=SpendingActionKind.REDUCE_TO,
        event_id="event_a",
        new_amount=Decimal("1"),
        series_event_ids=("event_a",),
        category="streaming",
        normal_amount=Decimal("10"),
        series_key="same",
    )
    other = SpendingAction(
        kind=SpendingActionKind.STOP,
        event_id="event_b",
        new_amount=None,
        series_event_ids=("event_b",),
        category="cloud_storage",
        normal_amount=Decimal("5"),
        series_key="other",
    )
    combos = action_combinations((stop, reduce, other))
    assert all(len({item.series_key for item in combo}) == len(combo) for combo in combos)


def test_not_recommended_fallback() -> None:
    state = _state(profile=_profile(current_available_balance=Decimal("1000")))
    request = _request(requested_amount=Decimal("4000"), desired_completion_date=date(2026, 3, 4))
    result = decide(state, request, (), CONFIG)
    assert result.recommended_payment_method is RecommendedPaymentMethod.NOT_RECOMMENDED
    assert result.affordability_status is AffordabilityStatus.NOT_AFFORDABLE
    assert result.payment_plan == ()
    assert result.spending_changes_needed == ()


def test_verifier_rejects_invalid_and_override_does_not_mutate() -> None:
    state = _state()
    request = _request()
    before = forecast_financial_state(state, CONFIG)
    bogus = CandidatePlan(
        method=RecommendedPaymentMethod.FULL_PAYMENT,
        affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
        payments=(),
        payment_option_id=None,
        spending_changes=(),
        total_paid=Decimal("0"),
        start_date=None,
        completion_date=None,
        completes_by_deadline=False,
        requires_spending_changes=False,
        is_preference_eligible=True,
        is_financially_safe=True,
        validation_failures=(),
    )
    checked = _validate(state, request, bogus)
    assert checked.validation_failures
    assert is_recommendable(checked) is False
    after = forecast_financial_state(state, CONFIG)
    assert before.all_entries == after.all_entries

    series = _streaming_series()
    stream_state = _state(recurring_series_candidates=(series,))
    baseline = forecast_financial_state(stream_state, CONFIG)
    stopped = forecast_financial_state(
        stream_state,
        CONFIG,
        recurrence_overrides=(RecurrenceOverride(event_id="event_s4", new_amount=None),),
    )
    assert any(entry.category == "streaming" for entry in baseline.all_entries)
    assert not any(
        entry.category == "streaming" and entry.is_generated_recurrence
        for entry in stopped.all_entries
    )
    again = forecast_financial_state(stream_state, CONFIG)
    assert again.all_entries == baseline.all_entries


def test_format_helpers() -> None:
    from decision.models import PlanPayment

    assert format_payment_plan(()) == "none"
    assert (
        format_payment_plan(
            (
                PlanPayment(date=date(2026, 1, 3), amount=Decimal("620.40")),
                PlanPayment(date=date(2026, 1, 15), amount=Decimal("10")),
            )
        )
        == "2026-01-03:620.4|2026-01-15:10"
    )
    assert format_spending_changes(()) == "none"
    action = SpendingAction(
        kind=SpendingActionKind.REDUCE_TO,
        event_id="event_x",
        new_amount=Decimal("23.50"),
        series_event_ids=("event_x",),
        category="streaming",
        normal_amount=Decimal("40"),
        series_key="s",
    )
    assert format_spending_changes((action,)) == "reduce_to:event_x:23.5"


def test_production_decision_code_quarantines_samples() -> None:
    root = Path("code/decision")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sample_requests" not in text
        assert "SampleDecision" not in text


def test_earliest_and_amount_are_preference_independent() -> None:
    request = _request(requested_amount=Decimal("8000"))
    full_state = _state()
    inst_state = _state(
        profile=_profile(
            payment_methods_user_will_consider=(ConsideredPaymentMethod.INSTALLMENTS,),
            max_installment_months=6,
        )
    )
    full_cap = compute_capacity(full_state, request, CONFIG)
    inst_cap = compute_capacity(inst_state, request, CONFIG)
    assert full_cap.amount_safe_to_pay == inst_cap.amount_safe_to_pay
    assert full_cap.earliest_date_for_full_payment == inst_cap.earliest_date_for_full_payment
    option = _option()
    full_dec = decide(full_state, request, (option,), CONFIG, capacity=full_cap)
    inst_dec = decide(inst_state, request, (option,), CONFIG, capacity=inst_cap)
    assert full_dec.amount_safe_to_pay == inst_dec.amount_safe_to_pay
    assert full_dec.earliest_date_for_full_payment == inst_dec.earliest_date_for_full_payment


def test_not_recommended_candidate_has_empty_plan() -> None:
    plan = not_recommended_candidate(_request())
    assert format_payment_plan(plan.payments) == "none"
    assert plan.method is RecommendedPaymentMethod.NOT_RECOMMENDED
