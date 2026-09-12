"""Structural dataset validation.

This module checks schema integrity, primary/foreign keys, and observed
file invariants. It does not evaluate affordability or payment eligibility.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from decimal import Decimal

from config import EXPECTED_BLANK_EVENT_AMOUNT_COUNT
from data.models import (
    Dataset,
    FinanceRequest,
    PaymentOption,
    PaymentOptionMethod,
)


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            "dataset validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        )


def _duplicate_ids(values: Iterable[str], label: str) -> list[str]:
    counts = Counter(values)
    return [
        f"duplicate {label} {item_id!r} ({count} rows)"
        for item_id, count in sorted(counts.items())
        if count > 1
    ]


def _request_amount(dataset: Dataset, request_id: str) -> Decimal | None:
    evaluation = dataset.evaluation_requests_by_id.get(request_id)
    if evaluation is not None:
        return evaluation.requested_amount
    sample = dataset.sample_requests_by_id.get(request_id)
    if sample is not None:
        return sample.request.requested_amount
    return None


def _validate_payment_option_structure(
    option: PaymentOption, requested_amount: Decimal | None
) -> list[str]:
    errors: list[str] = []
    prefix = f"payment option {option.payment_option_id}"
    if option.payment_amount < Decimal("0"):
        errors.append(f"{prefix}: payment_amount is negative")
    if option.financing_fee < Decimal("0"):
        errors.append(f"{prefix}: financing_fee is negative")
    if option.number_of_payments <= 0:
        errors.append(f"{prefix}: number_of_payments must be > 0")
    if option.total_payable_amount < Decimal("0"):
        errors.append(f"{prefix}: total_payable_amount is negative")

    if option.payment_method is PaymentOptionMethod.FULL_PAYMENT:
        if option.number_of_payments != 1:
            errors.append(f"{prefix}: full_payment number_of_payments must be 1")
        if option.payment_frequency_days is not None:
            errors.append(f"{prefix}: full_payment payment_frequency_days must be blank")
    elif option.payment_method is PaymentOptionMethod.INSTALLMENTS:
        if option.number_of_payments <= 1:
            errors.append(f"{prefix}: installments number_of_payments must be > 1")
        if option.payment_frequency_days is None:
            errors.append(f"{prefix}: installments payment_frequency_days is required")
        elif option.payment_frequency_days <= 0:
            errors.append(f"{prefix}: installments payment_frequency_days must be > 0")

    expected_total = option.payment_amount * option.number_of_payments
    if expected_total != option.total_payable_amount:
        errors.append(
            f"{prefix}: payment_amount * number_of_payments "
            f"({expected_total}) != total_payable_amount "
            f"({option.total_payable_amount})"
        )
    if requested_amount is not None:
        fee_total = requested_amount + option.financing_fee
        if fee_total != option.total_payable_amount:
            errors.append(
                f"{prefix}: requested_amount + financing_fee "
                f"({fee_total}) != total_payable_amount "
                f"({option.total_payable_amount})"
            )
    return errors


def collect_validation_errors(dataset: Dataset) -> list[str]:
    errors: list[str] = []

    errors.extend(_duplicate_ids((profile.user_id for profile in dataset.profiles), "user_id"))
    errors.extend(
        _duplicate_ids(
            (request.request_id for request in dataset.evaluation_requests),
            "evaluation request_id",
        )
    )
    errors.extend(
        _duplicate_ids(
            (sample.request_id for sample in dataset.sample_requests),
            "sample request_id",
        )
    )
    errors.extend(_duplicate_ids((event.event_id for event in dataset.events), "event_id"))
    errors.extend(
        _duplicate_ids(
            (option.payment_option_id for option in dataset.payment_options),
            "payment_option_id",
        )
    )
    errors.extend(
        _duplicate_ids((message.message_id for message in dataset.messages), "message_id")
    )
    errors.extend(_duplicate_ids((image.image_id for image in dataset.images), "image_id"))
    errors.extend(
        _duplicate_ids(
            (
                f"{rate.rate_date.isoformat()}|{rate.from_currency.value}|{rate.to_currency.value}"
                for rate in dataset.exchange_rates
            ),
            "exchange rate key",
        )
    )

    profile_ids = set(dataset.profiles_by_user_id)
    event_ids = set(dataset.events_by_id)
    request_ids = set(dataset.evaluation_requests_by_id) | set(
        dataset.sample_requests_by_id
    )

    def _check_user(user_id: str, context: str) -> None:
        if user_id not in profile_ids:
            errors.append(f"{context}: unknown user_id {user_id!r}")

    def _check_request(request_id: str, context: str) -> None:
        if request_id not in request_ids:
            errors.append(f"{context}: unknown request_id {request_id!r}")

    def _check_event(event_id: str, context: str) -> None:
        if event_id not in event_ids:
            errors.append(f"{context}: unknown event_id {event_id!r}")

    for request in dataset.evaluation_requests:
        _check_user(request.user_id, f"evaluation request {request.request_id}")
    for sample in dataset.sample_requests:
        _check_user(sample.user_id, f"sample request {sample.request_id}")
    for event in dataset.events:
        _check_user(event.user_id, f"financial event {event.event_id}")
        if event.linked_event_id is not None:
            _check_event(event.linked_event_id, f"financial event {event.event_id} linked_event_id")
    for option in dataset.payment_options:
        _check_request(option.request_id, f"payment option {option.payment_option_id}")
    for message in dataset.messages:
        _check_user(message.user_id, f"message {message.message_id}")
        if message.request_id is not None:
            _check_request(message.request_id, f"message {message.message_id}")
        if message.related_event_id is not None:
            _check_event(message.related_event_id, f"message {message.message_id}")
    for image in dataset.images:
        _check_user(image.user_id, f"image {image.image_id}")
        _check_request(image.request_id, f"image {image.image_id}")
        _check_event(image.related_event_id, f"image {image.image_id}")

    _check_non_negative_amounts(dataset, errors.append)
    _check_request_dates(dataset.evaluation_requests, errors.append, "evaluation request")
    _check_request_dates(
        (sample.request for sample in dataset.sample_requests),
        errors.append,
        "sample request",
    )

    blank_amount_events = [event for event in dataset.events if event.amount is None]
    if len(blank_amount_events) != EXPECTED_BLANK_EVENT_AMOUNT_COUNT:
        errors.append(
            "expected "
            f"{EXPECTED_BLANK_EVENT_AMOUNT_COUNT} blank event amounts, "
            f"found {len(blank_amount_events)}"
        )
    images_by_event = {image.related_event_id: image for image in dataset.images}
    for event in blank_amount_events:
        if event.event_id not in images_by_event:
            errors.append(
                f"financial event {event.event_id}: blank amount has no related image"
            )

    for image in dataset.images:
        if not image.path.is_file():
            errors.append(f"image {image.image_id}: file does not exist at {image.path}")

    for rate in dataset.exchange_rates:
        if rate.rate <= Decimal("0"):
            errors.append(
                f"exchange rate {rate.rate_date.isoformat()} "
                f"{rate.from_currency.value}->{rate.to_currency.value}: rate must be > 0"
            )

    for option in dataset.payment_options:
        errors.extend(
            _validate_payment_option_structure(
                option, _request_amount(dataset, option.request_id)
            )
        )

    return errors


def _check_non_negative_amounts(
    dataset: Dataset, add_error: Callable[[str], None]
) -> None:
    for profile in dataset.profiles:
        if profile.current_available_balance < Decimal("0"):
            add_error(f"profile {profile.user_id}: current_available_balance is negative")
        if profile.minimum_balance_to_keep < Decimal("0"):
            add_error(f"profile {profile.user_id}: minimum_balance_to_keep is negative")
    for request in dataset.evaluation_requests:
        if request.requested_amount < Decimal("0"):
            add_error(f"evaluation request {request.request_id}: requested_amount is negative")
    for sample in dataset.sample_requests:
        if sample.request.requested_amount < Decimal("0"):
            add_error(f"sample request {sample.request_id}: requested_amount is negative")
        if sample.decision.amount_safe_to_pay < Decimal("0"):
            add_error(f"sample request {sample.request_id}: amount_safe_to_pay is negative")
    for event in dataset.events:
        if event.amount is not None and event.amount < Decimal("0"):
            add_error(f"financial event {event.event_id}: amount is negative")
        if (
            event.minimum_allowed_amount is not None
            and event.minimum_allowed_amount < Decimal("0")
        ):
            add_error(
                f"financial event {event.event_id}: minimum_allowed_amount is negative"
            )


def _check_request_dates(
    requests: Iterable[FinanceRequest],
    add_error: Callable[[str], None],
    label: str,
) -> None:
    for request in requests:
        if request.desired_completion_date < request.request_date:
            add_error(
                f"{label} {request.request_id}: "
                "desired_completion_date is before request_date"
            )


def validate_dataset(dataset: Dataset) -> None:
    errors = collect_validation_errors(dataset)
    if errors:
        raise DatasetValidationError(errors)
