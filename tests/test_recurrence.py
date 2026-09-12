from __future__ import annotations

from datetime import date
from decimal import Decimal

from data.models import (
    ConsideredPaymentMethod,
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinancialProfile,
    Flexibility,
)
from finance.models import (
    AmountBehavior,
    Cadence,
    CadenceConfidence,
    LifecycleRole,
    NormalizedCashEvent,
)
from finance.recurrence import (
    build_recurring_series_candidates,
    infer_cadence,
    normalize_description,
)


def _profile(**overrides: object) -> FinancialProfile:
    values: dict[str, object] = {
        "user_id": "user_01",
        "home_currency": Currency.ZAR,
        "current_available_balance": Decimal("100"),
        "minimum_balance_to_keep": Decimal("10"),
        "financial_priorities": ("education",),
        "expense_categories_to_protect": ("rent",),
        "expense_categories_user_is_willing_to_reduce": ("dining",),
        "expense_categories_user_is_willing_to_stop": ("streaming",),
        "payment_methods_user_will_consider": (ConsideredPaymentMethod.FULL_PAYMENT,),
        "max_installment_months": None,
    }
    values.update(overrides)
    return FinancialProfile(**values)  # type: ignore[arg-type]


def _hist(
    event_id: str,
    cash_date: date,
    amount: Decimal | None,
    *,
    category: str = "rent",
    description: str = "Apartment rent transfer",
    direction: EventDirection = EventDirection.DEBIT,
    event_type: EventType = EventType.EXPENSE,
    flexibility: Flexibility = Flexibility.FIXED,
    minimum_allowed_amount: Decimal | None = None,
) -> NormalizedCashEvent:
    return NormalizedCashEvent(
        source_event_id=event_id,
        user_id="user_01",
        cash_date=cash_date,
        event_date=cash_date,
        settlement_date=cash_date,
        direction=direction,
        amount_home_currency=amount,
        original_amount=amount,
        original_currency=Currency.ZAR,
        event_type=event_type,
        category=category,
        description=description,
        status=EventStatus.SETTLED,
        lifecycle_role=LifecycleRole.STANDALONE,
        lifecycle_type=None,
        flexibility=flexibility,
        minimum_allowed_amount=minimum_allowed_amount,
        is_historical=True,
        is_prospective=False,
        counts_as_cash=amount is not None,
        ignore_reason=None,
        requires_external_evidence=amount is None,
    )


def test_description_normalization_is_minimal() -> None:
    assert normalize_description("  Apartment   Rent Transfer ") == "apartment rent transfer"
    assert normalize_description("Netflix") != normalize_description("Amazon Prime")


def test_weekly_biweekly_monthly_and_irregular_cadence() -> None:
    weekly = infer_cadence(
        (date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15), date(2024, 1, 22), date(2024, 1, 29))
    )
    assert weekly[0] is Cadence.WEEKLY
    assert weekly[2] is CadenceConfidence.HIGH

    biweekly = infer_cadence(
        (date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 29), date(2024, 2, 12))
    )
    assert biweekly[0] is Cadence.BIWEEKLY
    assert biweekly[2] is CadenceConfidence.HIGH

    monthly = infer_cadence(
        (date(2024, 1, 2), date(2024, 2, 2), date(2024, 3, 2), date(2024, 4, 2))
    )
    assert monthly[0] is Cadence.MONTHLY
    assert monthly[2] is CadenceConfidence.HIGH

    irregular = infer_cadence((date(2024, 1, 1), date(2024, 3, 20), date(2024, 8, 9)))
    assert irregular[0] is Cadence.IRREGULAR
    assert irregular[2] is CadenceConfidence.LOW

    one = infer_cadence((date(2024, 1, 1),))
    assert one[0] is Cadence.UNKNOWN
    assert one[2] is CadenceConfidence.LOW


def test_fixed_and_variable_amount_series() -> None:
    profile = _profile()
    fixed = (
        _hist("a", date(2024, 1, 2), Decimal("100")),
        _hist("b", date(2024, 2, 2), Decimal("100")),
        _hist("c", date(2024, 3, 2), Decimal("100")),
        _hist("d", date(2024, 4, 2), Decimal("100")),
    )
    variable = (
        _hist(
            "e",
            date(2024, 1, 7),
            Decimal("10"),
            category="groceries",
            description="Weekly groceries",
        ),
        _hist(
            "f",
            date(2024, 1, 14),
            Decimal("18"),
            category="groceries",
            description="Weekly groceries",
        ),
        _hist(
            "g",
            date(2024, 1, 21),
            Decimal("12"),
            category="groceries",
            description="Weekly groceries",
        ),
        _hist(
            "h",
            date(2024, 1, 28),
            Decimal("15"),
            category="groceries",
            description="Weekly groceries",
        ),
    )
    candidates = build_recurring_series_candidates(
        profile=profile,
        historical_events=fixed + variable,
        confirmed_scheduled_income=(),
        requires_message_resolution=False,
    )
    rent = next(item for item in candidates if item.category == "rent")
    groceries = next(item for item in candidates if item.category == "groceries")
    assert rent.amount_behavior is AmountBehavior.FIXED
    assert rent.representative_amount == Decimal("100")
    assert groceries.amount_behavior is AmountBehavior.VARIABLE
    assert groceries.representative_amount is None
    assert groceries.observed_amounts_home_currency == (
        Decimal("10"),
        Decimal("18"),
        Decimal("12"),
        Decimal("15"),
    )


def test_one_off_and_sparse_flexible_candidate() -> None:
    profile = _profile()
    one_off = _hist(
        "shop",
        date(2024, 2, 1),
        Decimal("80"),
        category="shopping",
        description="One-off gadget",
    )
    sparse = (
        _hist(
            "d1",
            date(2024, 12, 18),
            Decimal("116"),
            category="dining",
            description="Weekend food delivery",
            flexibility=Flexibility.REDUCIBLE,
            minimum_allowed_amount=Decimal("66"),
        ),
        _hist(
            "d2",
            date(2025, 4, 23),
            Decimal("90"),
            category="dining",
            description="Weekend food delivery",
            flexibility=Flexibility.REDUCIBLE,
            minimum_allowed_amount=Decimal("66"),
        ),
    )
    candidates = build_recurring_series_candidates(
        profile=profile,
        historical_events=(one_off, *sparse),
        confirmed_scheduled_income=(),
        requires_message_resolution=False,
    )
    shopping = next(item for item in candidates if item.category == "shopping")
    dining = next(item for item in candidates if item.category == "dining")
    assert shopping.likely_recurring is False
    assert dining.likely_recurring is True
    assert dining.cadence_confidence is CadenceConfidence.LOW
    assert dining.inferred_cadence is Cadence.IRREGULAR
    assert dining.profile_allows_reduce is True
    assert dining.minimum_allowed_amount == Decimal("66")


def test_subscription_sparse_is_still_likely() -> None:
    profile = _profile()
    events = (
        _hist(
            "s1",
            date(2024, 1, 10),
            Decimal("19"),
            category="streaming",
            description="Family streaming plan",
            event_type=EventType.SUBSCRIPTION,
            flexibility=Flexibility.STOPPABLE,
        ),
        _hist(
            "s2",
            date(2024, 2, 10),
            Decimal("19"),
            category="streaming",
            description="Family streaming plan",
            event_type=EventType.SUBSCRIPTION,
            flexibility=Flexibility.STOPPABLE,
        ),
    )
    candidates = build_recurring_series_candidates(
        profile=profile,
        historical_events=events,
        confirmed_scheduled_income=(),
        requires_message_resolution=False,
    )
    streaming = candidates[0]
    assert streaming.likely_recurring is True
    assert streaming.profile_allows_stop is True
    assert streaming.inferred_cadence is Cadence.MONTHLY
