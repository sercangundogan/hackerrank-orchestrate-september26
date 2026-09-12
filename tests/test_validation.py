from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from config import EXPECTED_BLANK_EVENT_AMOUNT_COUNT
from data.loader import assemble_dataset
from data.models import (
    AffordabilityStatus,
    ConsideredPaymentMethod,
    Currency,
    Dataset,
    EventDirection,
    EventStatus,
    EventType,
    ExchangeRate,
    FinanceRequest,
    FinancialEvent,
    FinancialProfile,
    Flexibility,
    ImageReference,
    Message,
    MessageSourceType,
    OutputTemplateRow,
    PaymentOption,
    PaymentOptionMethod,
    RecommendedPaymentMethod,
    RequestType,
    SampleDecision,
    SampleFinanceRequest,
)
from data.validation import collect_validation_errors, validate_dataset


def _profile(user_id: str = "user_01") -> FinancialProfile:
    return FinancialProfile(
        user_id=user_id,
        home_currency=Currency.ZAR,
        current_available_balance=Decimal("100"),
        minimum_balance_to_keep=Decimal("10"),
        financial_priorities=("education",),
        expense_categories_to_protect=("rent",),
        expense_categories_user_is_willing_to_reduce=("dining",),
        expense_categories_user_is_willing_to_stop=("streaming",),
        payment_methods_user_will_consider=(ConsideredPaymentMethod.FULL_PAYMENT,),
        max_installment_months=None,
    )


def _request(
    request_id: str = "request_01",
    user_id: str = "user_01",
    amount: Decimal = Decimal("50"),
) -> FinanceRequest:
    return FinanceRequest(
        request_id=request_id,
        user_id=user_id,
        request_date=date(2024, 3, 3),
        request_type=RequestType.PURCHASE,
        requested_amount=amount,
        desired_completion_date=date(2024, 3, 20),
        allows_partial_payment=True,
        request_text="test",
    )


def _sample(request: FinanceRequest) -> SampleFinanceRequest:
    return SampleFinanceRequest(
        request=request,
        decision=SampleDecision(
            amount_safe_to_pay=request.requested_amount,
            affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
            recommended_payment_method=RecommendedPaymentMethod.FULL_PAYMENT,
            payment_plan="none",
            earliest_date_for_full_payment=request.request_date,
            spending_changes_needed="none",
            decision_explanation="test",
        ),
    )


def _event(
    event_id: str = "event_01",
    user_id: str = "user_01",
    amount: Decimal | None = Decimal("10"),
    linked_event_id: str | None = None,
) -> FinancialEvent:
    return FinancialEvent(
        event_id=event_id,
        user_id=user_id,
        event_type=EventType.EXPENSE,
        description="test",
        category="rent",
        direction=EventDirection.DEBIT,
        amount=amount,
        currency=Currency.ZAR,
        event_date=date(2024, 1, 1),
        settlement_date=date(2024, 1, 1),
        status=EventStatus.SETTLED,
        linked_event_id=linked_event_id,
        flexibility=Flexibility.FIXED,
        minimum_allowed_amount=None,
    )


def _option(
    option_id: str = "payment_option_01",
    request_id: str = "request_01",
    method: PaymentOptionMethod = PaymentOptionMethod.FULL_PAYMENT,
    payment_amount: Decimal = Decimal("50"),
    number_of_payments: int = 1,
    frequency: int | None = None,
    fee: Decimal = Decimal("0"),
    total: Decimal = Decimal("50"),
) -> PaymentOption:
    return PaymentOption(
        payment_option_id=option_id,
        request_id=request_id,
        payment_method=method,
        payment_amount=payment_amount,
        number_of_payments=number_of_payments,
        first_payment_date=date(2024, 3, 3),
        payment_frequency_days=frequency,
        financing_fee=fee,
        total_payable_amount=total,
    )


def _image(
    image_id: str = "image_01",
    related_event_id: str = "event_blank",
    path: Path | None = None,
) -> ImageReference:
    return ImageReference(
        image_id=image_id,
        user_id="user_01",
        request_id="request_01",
        related_event_id=related_event_id,
        path=path or Path("/tmp/missing-image.png"),
    )


def _dataset(
    *,
    profiles: tuple[FinancialProfile, ...] | None = None,
    evaluation_requests: tuple[FinanceRequest, ...] = (),
    sample_requests: tuple[SampleFinanceRequest, ...] = (),
    events: tuple[FinancialEvent, ...] = (),
    exchange_rates: tuple[ExchangeRate, ...] = (),
    payment_options: tuple[PaymentOption, ...] = (),
    messages: tuple[Message, ...] = (),
    images: tuple[ImageReference, ...] = (),
    output_template_rows: tuple[OutputTemplateRow, ...] = (),
) -> Dataset:
    if profiles is None:
        profiles = (_profile(),)
    return assemble_dataset(
        profiles=profiles,
        evaluation_requests=evaluation_requests,
        sample_requests=sample_requests,
        events=events,
        exchange_rates=exchange_rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
        output_template_rows=output_template_rows,
    )


def test_real_dataset_passes_structural_validation(dataset: Dataset) -> None:
    validate_dataset(dataset)
    blank_events = [event for event in dataset.events if event.amount is None]
    assert len(blank_events) == EXPECTED_BLANK_EVENT_AMOUNT_COUNT
    linked = [event for event in dataset.events if event.linked_event_id is not None]
    assert len(linked) == 58
    for event in linked:
        assert event.linked_event_id in dataset.events_by_id


def test_primary_key_uniqueness() -> None:
    errors = collect_validation_errors(
        _dataset(profiles=(_profile("user_01"), _profile("user_01")))
    )
    assert any("duplicate user_id" in error for error in errors)

    errors = collect_validation_errors(
        _dataset(
            evaluation_requests=(
                _request("request_26"),
                _request("request_26"),
            )
        )
    )
    assert any("duplicate evaluation request_id" in error for error in errors)


def test_foreign_key_integrity() -> None:
    errors = collect_validation_errors(
        _dataset(evaluation_requests=(_request(user_id="user_missing"),))
    )
    assert any("unknown user_id 'user_missing'" in error for error in errors)

    errors = collect_validation_errors(
        _dataset(payment_options=(_option(request_id="request_missing"),))
    )
    assert any("unknown request_id 'request_missing'" in error for error in errors)

    errors = collect_validation_errors(
        _dataset(
            events=(_event(linked_event_id="event_missing"),),
        )
    )
    assert any("unknown event_id 'event_missing'" in error for error in errors)


def test_linked_event_resolution_on_real_data(dataset: Dataset) -> None:
    child = dataset.events_by_id["event_99"]
    parent = dataset.events_by_id[child.linked_event_id or ""]
    assert parent.event_id == "event_98"


def test_blank_amount_image_consistency_on_real_data(dataset: Dataset) -> None:
    blank_ids = {event.event_id for event in dataset.events if event.amount is None}
    image_event_ids = {image.related_event_id for image in dataset.images}
    assert blank_ids == image_event_ids
    for image in dataset.images:
        assert image.path.is_file()


def test_blank_amount_without_image_is_reported() -> None:
    blank_events = tuple(
        _event(event_id=f"event_blank_{index}", amount=None)
        for index in range(EXPECTED_BLANK_EVENT_AMOUNT_COUNT)
    )
    errors = collect_validation_errors(_dataset(events=blank_events))
    assert any("blank amount has no related image" in error for error in errors)


def test_missing_image_file_is_reported(tmp_path: Path) -> None:
    blank_events = tuple(
        _event(event_id=f"event_blank_{index}", amount=None)
        for index in range(EXPECTED_BLANK_EVENT_AMOUNT_COUNT)
    )
    images = tuple(
        _image(
            image_id=f"image_{index:02d}",
            related_event_id=f"event_blank_{index}",
            path=tmp_path / f"image_{index:02d}.png",
        )
        for index in range(EXPECTED_BLANK_EVENT_AMOUNT_COUNT)
    )
    errors = collect_validation_errors(_dataset(events=blank_events, images=images))
    assert any("file does not exist" in error for error in errors)


def test_payment_option_structural_arithmetic() -> None:
    request = _request(amount=Decimal("100"))
    good = _option(
        method=PaymentOptionMethod.INSTALLMENTS,
        payment_amount=Decimal("55"),
        number_of_payments=2,
        frequency=30,
        fee=Decimal("10"),
        total=Decimal("110"),
    )
    assert not any(
        "payment_amount * number_of_payments" in error
        for error in collect_validation_errors(
            _dataset(sample_requests=(_sample(request),), payment_options=(good,))
        )
        if "payment_option_01" in error
    )

    bad_total = _option(
        method=PaymentOptionMethod.INSTALLMENTS,
        payment_amount=Decimal("55"),
        number_of_payments=2,
        frequency=30,
        fee=Decimal("10"),
        total=Decimal("999"),
    )
    errors = collect_validation_errors(
        _dataset(sample_requests=(_sample(request),), payment_options=(bad_total,))
    )
    assert any("payment_amount * number_of_payments" in error for error in errors)
    assert any("requested_amount + financing_fee" in error for error in errors)


def test_full_payment_and_installment_shape_rules() -> None:
    request = _request()
    sample = _sample(request)
    full_with_freq = _option(frequency=30)
    errors = collect_validation_errors(
        _dataset(sample_requests=(sample,), payment_options=(full_with_freq,))
    )
    assert any("full_payment payment_frequency_days must be blank" in error for error in errors)

    installment_without_freq = _option(
        method=PaymentOptionMethod.INSTALLMENTS,
        payment_amount=Decimal("25"),
        number_of_payments=2,
        frequency=None,
        fee=Decimal("0"),
        total=Decimal("50"),
    )
    errors = collect_validation_errors(
        _dataset(sample_requests=(sample,), payment_options=(installment_without_freq,))
    )
    assert any(
        "installments payment_frequency_days is required" in error for error in errors
    )


def test_desired_completion_date_not_before_request_date() -> None:
    request = FinanceRequest(
        request_id="request_01",
        user_id="user_01",
        request_date=date(2024, 3, 20),
        request_type=RequestType.PURCHASE,
        requested_amount=Decimal("10"),
        desired_completion_date=date(2024, 3, 3),
        allows_partial_payment=False,
        request_text="test",
    )
    errors = collect_validation_errors(_dataset(evaluation_requests=(request,)))
    assert any("desired_completion_date is before request_date" in error for error in errors)


def test_message_optional_foreign_keys() -> None:
    message = Message(
        message_id="message_01",
        user_id="user_01",
        request_id="request_missing",
        related_event_id="event_missing",
        sent_at=datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc),
        source_type=MessageSourceType.EMPLOYER,
        message_text="test",
    )
    errors = collect_validation_errors(_dataset(messages=(message,)))
    assert any("unknown request_id 'request_missing'" in error for error in errors)
    assert any("unknown event_id 'event_missing'" in error for error in errors)
