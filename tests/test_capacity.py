from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from data.models import (
    ConsideredPaymentMethod,
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinanceRequest,
    RequestType,
)
from decision.capacity import (
    amount_safe_to_pay,
    compute_capacity,
    earliest_date_for_full_payment,
    infer_money_quantum,
    is_payment_safe,
)
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
)
from evidence.resolver import resolve_financial_state
from finance.adjustments import ForecastAdjustment, ForecastAdjustmentKind
from finance.forecast import forecast_financial_state
from finance.forecast_models import ForecastConfig, UnresolvedForecastError
from finance.models import Cadence, IgnoreReason
from tests.factories import FakeRateLookup
from tests.test_forecast import _event, _profile, _series, _state


def _request(**overrides: object) -> FinanceRequest:
    values: dict[str, object] = {
        "request_id": "request_test",
        "user_id": "user_test",
        "request_date": date(2026, 3, 3),
        "request_type": RequestType.PURCHASE,
        "requested_amount": Decimal("1000"),
        "desired_completion_date": date(2026, 3, 10),
        "allows_partial_payment": True,
        "request_text": "test purchase",
    }
    values.update(overrides)
    return FinanceRequest(**values)  # type: ignore[arg-type]


def test_infer_quantum_from_source_precision() -> None:
    assert infer_money_quantum(Decimal("25256"), Decimal("18000")) == Decimal("1")
    assert infer_money_quantum(Decimal("33.50"), Decimal("1000")) == Decimal("0.01")
    assert infer_money_quantum(Decimal("10.5"), Decimal("1.25")) == Decimal("0.01")


def test_zero_capacity_when_already_at_floor() -> None:
    state = _state(profile=_profile(current_available_balance=Decimal("1000")))
    request = _request(requested_amount=Decimal("400"))
    result = compute_capacity(state, request)
    assert result.amount_safe_to_pay == Decimal("0")
    assert result.full_payment_safe_today is False


def test_partial_capacity() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("4500"))
    result = compute_capacity(state, request)
    assert result.amount_safe_to_pay == Decimal("4000")
    assert Decimal("0") < result.amount_safe_to_pay < request.requested_amount


def test_full_requested_amount_safe() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("2000"))
    result = compute_capacity(state, request)
    assert result.amount_safe_to_pay == Decimal("2000")
    assert result.full_payment_safe_today is True
    assert result.earliest_date_for_full_payment == request.request_date


def test_cap_at_requested_amount() -> None:
    state = _state(profile=_profile(current_available_balance=Decimal("20000")))
    request = _request(requested_amount=Decimal("150"))
    result = compute_capacity(state, request)
    assert result.amount_safe_to_pay == Decimal("150")
    assert result.amount_safe_to_pay <= request.requested_amount


def test_exact_minimum_balance_equality_is_safe() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("4000"))
    exact = is_payment_safe(state, request, request.request_date, Decimal("4000"))
    assert exact.is_safe
    assert exact.minimum_observed_balance == Decimal("1000")


def test_one_quantum_above_maximum_is_unsafe() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("4000.00"))
    result = compute_capacity(state, request)
    assert result.amount_safe_to_pay == Decimal("4000.00")
    above = is_payment_safe(state, request, request.request_date, Decimal("4000.01"))
    assert above.is_safe is False


def test_decimal_precision_uses_request_quantum() -> None:
    state = _state(profile=_profile(current_available_balance=Decimal("1050.25")))
    request = _request(requested_amount=Decimal("80.40"))
    result = compute_capacity(state, request)
    assert result.quantum == Decimal("0.01")
    assert result.amount_safe_to_pay == Decimal("50.25")
    assert is_payment_safe(
        state, request, request.request_date, result.amount_safe_to_pay
    ).is_safe
    assert not is_payment_safe(
        state, request, request.request_date, result.amount_safe_to_pay + Decimal("0.01")
    ).is_safe


def test_pending_debit_reduces_capacity() -> None:
    pending = _event(amount_home_currency=Decimal("1500"))
    state = _state(pending_debits=(pending,), classified_events=(pending,))
    request = _request(requested_amount=Decimal("4000"))
    without = compute_capacity(_state(), request)
    with_pending = compute_capacity(state, request)
    assert with_pending.amount_safe_to_pay == Decimal("2500")
    assert with_pending.amount_safe_to_pay < without.amount_safe_to_pay


def test_resolved_image_obligation_reduces_capacity() -> None:
    blank = _event(
        source_event_id="event_blank",
        amount_home_currency=None,
        original_amount=None,
        status=EventStatus.SCHEDULED,
        cash_date=date(2026, 3, 3),
        settlement_date=date(2026, 3, 3),
        event_date=date(2026, 3, 3),
        requires_external_evidence=True,
        is_prospective=False,
        counts_as_cash=False,
        ignore_reason=IgnoreReason.BLANK_AMOUNT,
        description="Outstanding bill",
        category="utilities",
    )
    state = _state(
        classified_events=(blank,),
        unresolved_events=(blank,),
        ignored_events=(blank,),
    )
    fact = EvidenceFact(
        source_type=EvidenceSourceType.IMAGE,
        source_id="image_x",
        user_id="user_test",
        request_id="request_test",
        related_event_id="event_blank",
        fact_type=FactType.EVENT_AMOUNT,
        amount=Decimal("800"),
        currency=Currency.ZAR,
        effective_date=None,
        status=FactStatus.ACTIVE,
        supersedes_event_id="event_blank",
        confidence=EvidenceConfidence.HIGH,
        extraction_method=ExtractionMethod.VLM,
        raw_reference="image_x",
        notes="balance due",
    )
    resolved = resolve_financial_state(state, (fact,), repository=FakeRateLookup())
    request = _request(requested_amount=Decimal("4000"))
    before = compute_capacity(_state(), request)
    after = compute_capacity(resolved, request)
    assert after.amount_safe_to_pay == Decimal("3200")
    assert after.amount_safe_to_pay < before.amount_safe_to_pay


def test_future_salary_raise_does_not_inflate_today_capacity() -> None:
    series = _series(
        category="salary",
        normalized_description="payroll credit",
        original_description="Payroll credit",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        inferred_cadence_days=30,
        observed_dates=(
            date(2025, 12, 15),
            date(2026, 1, 15),
            date(2026, 2, 15),
        ),
        event_ids=("s1", "s2", "s3"),
        representative_amount=Decimal("2000"),
        observed_amounts_home_currency=(Decimal("2000"),) * 3,
    )
    base = _state(recurring_series_candidates=(series,))
    raised = _state(
        recurring_series_candidates=(series,),
        forecast_adjustments=(
            ForecastAdjustment(
                kind=ForecastAdjustmentKind.SALARY_AMOUNT,
                amount_home=Decimal("8000"),
                effective_date=date(2026, 3, 15),
                source_ids=("message_raise",),
                notes="raise",
            ),
        ),
    )
    request = _request(requested_amount=Decimal("4500"))
    assert amount_safe_to_pay(base, request) == amount_safe_to_pay(raised, request)


def test_monotonicity_larger_payment_stays_unsafe() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("5000"))
    unsafe = Decimal("4000.01")
    assert not is_payment_safe(state, request, request.request_date, unsafe).is_safe
    assert not is_payment_safe(
        state, request, request.request_date, unsafe + Decimal("100")
    ).is_safe
    safe = Decimal("4000")
    assert is_payment_safe(state, request, request.request_date, safe).is_safe


def test_earliest_is_request_date_when_full_safe_today() -> None:
    result = compute_capacity(_state(), _request(requested_amount=Decimal("10")))
    assert result.earliest_date_for_full_payment == date(2026, 3, 3)


def test_earliest_full_payment_on_payday() -> None:
    salary = _event(
        source_event_id="sal_1",
        cash_date=date(2026, 3, 10),
        event_date=date(2026, 3, 10),
        settlement_date=date(2026, 3, 10),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("500"),
        event_type=EventType.INCOME,
        category="salary",
        description="Confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    state = _state(
        profile=_profile(current_available_balance=Decimal("1000")),
        confirmed_scheduled_income=(salary,),
        classified_events=(salary,),
    )
    request = _request(requested_amount=Decimal("400"))
    result = compute_capacity(state, request)
    assert result.amount_safe_to_pay == Decimal("0")
    assert result.earliest_date_for_full_payment == date(2026, 3, 10)
    payday = is_payment_safe(state, request, date(2026, 3, 10), Decimal("400"))
    assert payday.is_safe
    kinds = [
        entry.kind.value
        for day in payday.forecast.daily_forecasts
        if day.date == date(2026, 3, 10)
        for entry in day.entries
    ]
    assert kinds[0] == "confirmed_credit"
    assert kinds[-1] == "candidate_payment"


def test_earliest_chooses_first_safe_income_date() -> None:
    small = _event(
        source_event_id="sal_small",
        cash_date=date(2026, 3, 8),
        event_date=date(2026, 3, 8),
        settlement_date=date(2026, 3, 8),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("50"),
        event_type=EventType.INCOME,
        category="salary",
        description="Small confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    large = _event(
        source_event_id="sal_large",
        cash_date=date(2026, 3, 20),
        event_date=date(2026, 3, 20),
        settlement_date=date(2026, 3, 20),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("2000"),
        event_type=EventType.INCOME,
        category="salary",
        description="Large confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    state = _state(
        profile=_profile(current_available_balance=Decimal("1000")),
        confirmed_scheduled_income=(small, large),
        classified_events=(small, large),
    )
    request = _request(requested_amount=Decimal("500"))
    result = compute_capacity(state, request)
    assert result.earliest_date_for_full_payment == date(2026, 3, 20)


def test_earliest_may_be_after_desired_completion_date() -> None:
    salary = _event(
        source_event_id="sal_1",
        cash_date=date(2026, 3, 20),
        event_date=date(2026, 3, 20),
        settlement_date=date(2026, 3, 20),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("2000"),
        event_type=EventType.INCOME,
        category="salary",
        description="Confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    state = _state(
        profile=_profile(current_available_balance=Decimal("1000")),
        confirmed_scheduled_income=(salary,),
        classified_events=(salary,),
    )
    request = _request(
        requested_amount=Decimal("500"),
        desired_completion_date=date(2026, 3, 5),
    )
    result = compute_capacity(state, request)
    assert result.earliest_date_for_full_payment is not None
    assert result.earliest_date_for_full_payment > request.desired_completion_date


def test_no_safe_date_returns_none() -> None:
    state = _state(profile=_profile(current_available_balance=Decimal("1000")))
    request = _request(requested_amount=Decimal("50000"))
    result = compute_capacity(state, request)
    assert result.earliest_date_for_full_payment is None
    assert result.amount_safe_to_pay == Decimal("0")


def test_employment_end_prevents_false_future_safe_date() -> None:
    series = _series(
        category="salary",
        normalized_description="payroll credit",
        original_description="Payroll credit",
        direction=EventDirection.CREDIT,
        event_type=EventType.INCOME,
        inferred_cadence=Cadence.MONTHLY,
        inferred_cadence_days=30,
        observed_dates=(
            date(2025, 12, 15),
            date(2026, 1, 15),
            date(2026, 2, 15),
        ),
        event_ids=("s1", "s2", "s3"),
        representative_amount=Decimal("5000"),
        observed_amounts_home_currency=(Decimal("5000"),) * 3,
    )
    request = _request(requested_amount=Decimal("4500"))
    with_salary = compute_capacity(
        _state(
            profile=_profile(current_available_balance=Decimal("1000")),
            recurring_series_candidates=(series,),
        ),
        request,
    )
    ended = compute_capacity(
        _state(
            profile=_profile(current_available_balance=Decimal("1000")),
            recurring_series_candidates=(series,),
            forecast_adjustments=(
                ForecastAdjustment(
                    kind=ForecastAdjustmentKind.STOP_SALARY_PROJECTION,
                    source_ids=("message_end",),
                    notes="employment ended",
                ),
            ),
        ),
        request,
    )
    assert with_salary.earliest_date_for_full_payment is not None
    assert ended.earliest_date_for_full_payment is None


def test_scheduled_obligation_can_delay_earliest_date() -> None:
    salary = _event(
        source_event_id="sal_1",
        cash_date=date(2026, 3, 10),
        event_date=date(2026, 3, 10),
        settlement_date=date(2026, 3, 10),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("800"),
        event_type=EventType.INCOME,
        category="salary",
        description="Confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    rent = _event(
        source_event_id="rent_1",
        cash_date=date(2026, 3, 10),
        event_date=date(2026, 3, 10),
        settlement_date=date(2026, 3, 10),
        amount_home_currency=Decimal("700"),
        category="rent",
        description="Scheduled rent",
        status=EventStatus.SCHEDULED,
    )
    later_ok = _event(
        source_event_id="sal_2",
        cash_date=date(2026, 3, 25),
        event_date=date(2026, 3, 25),
        settlement_date=date(2026, 3, 25),
        direction=EventDirection.CREDIT,
        amount_home_currency=Decimal("2000"),
        event_type=EventType.INCOME,
        category="salary",
        description="Later confirmed salary",
        status=EventStatus.SCHEDULED,
    )
    state = _state(
        profile=_profile(current_available_balance=Decimal("1000")),
        confirmed_scheduled_income=(salary, later_ok),
        scheduled_debits=(rent,),
        classified_events=(salary, rent, later_ok),
    )
    request = _request(requested_amount=Decimal("500"))
    result = compute_capacity(state, request)
    assert result.earliest_date_for_full_payment == date(2026, 3, 25)
    assert not is_payment_safe(state, request, date(2026, 3, 10), Decimal("500")).is_safe


def test_preferences_do_not_change_capacity() -> None:
    request_full = _request(allows_partial_payment=False)
    request_partial = _request(allows_partial_payment=True)
    reject_full = _state(
        profile=_profile(
            payment_methods_user_will_consider=(ConsideredPaymentMethod.INSTALLMENTS,),
            max_installment_months=6,
        )
    )
    allow_full = _state(
        profile=_profile(
            payment_methods_user_will_consider=(
                ConsideredPaymentMethod.FULL_PAYMENT,
                ConsideredPaymentMethod.PARTIAL_PAYMENT,
            )
        )
    )
    a = compute_capacity(reject_full, request_full)
    b = compute_capacity(allow_full, request_partial)
    assert a.amount_safe_to_pay == b.amount_safe_to_pay
    assert a.earliest_date_for_full_payment == b.earliest_date_for_full_payment


def test_strict_unresolved_state_raises() -> None:
    blank = _event(
        source_event_id="event_blank",
        amount_home_currency=None,
        original_amount=None,
        requires_external_evidence=True,
        is_prospective=True,
        counts_as_cash=False,
        ignore_reason=IgnoreReason.BLANK_AMOUNT,
    )
    state = _state(unresolved_events=(blank,), classified_events=(blank,))
    with pytest.raises(UnresolvedForecastError):
        compute_capacity(
            state,
            _request(),
            ForecastConfig(strict_unresolved_amounts=True),
        )


def test_nonstrict_unresolved_preserves_diagnostics() -> None:
    blank = _event(
        source_event_id="event_blank",
        amount_home_currency=None,
        original_amount=None,
        requires_external_evidence=True,
        is_prospective=True,
        counts_as_cash=False,
        ignore_reason=IgnoreReason.BLANK_AMOUNT,
    )
    state = _state(unresolved_events=(blank,), classified_events=(blank,))
    result = compute_capacity(
        state,
        _request(),
        ForecastConfig(strict_unresolved_amounts=False),
    )
    assert result.unresolved_reasons
    assert any("event_blank" in item for item in result.unresolved_reasons)


def test_capacity_does_not_mutate_state() -> None:
    state = _state()
    request = _request(requested_amount=Decimal("2000"))
    before = forecast_financial_state(state)
    compute_capacity(state, request)
    after = forecast_financial_state(state)
    assert before.all_entries == after.all_entries
    assert before.minimum_observed_balance == after.minimum_observed_balance


def test_sample_evaluation_helper_is_quarantined() -> None:
    text = Path("code/decision/capacity.py").read_text(encoding="utf-8")
    assert "sample_requests" not in text
    assert "SampleDecision" not in text
    assert "amount_safe_to_pay" in Path("tests/capacity_evaluation.py").read_text(
        encoding="utf-8"
    )


def test_capacity_module_has_no_decision_policy() -> None:
    text = Path("code/decision/capacity.py").read_text(encoding="utf-8")
    assert "affordability_status" not in text
    assert "recommended_payment_method" not in text
    assert "installment" not in text.lower()
    assert "spending_changes" not in text
