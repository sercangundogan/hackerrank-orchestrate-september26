"""UTF-8 CSV loaders that convert participant files into typed domain models."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import TypeVar

from config import DatasetPaths
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

EnumT = TypeVar("EnumT", bound=Enum)


class ParseError(ValueError):
    """Raised when a CSV cell cannot be converted to the expected type."""


def _filename(path: Path) -> str:
    return path.name


def _require_columns(path: Path, fieldnames: Sequence[str] | None, required: Sequence[str]) -> None:
    present = list(fieldnames or [])
    missing = [column for column in required if column not in present]
    if missing:
        raise ParseError(
            f"{_filename(path)} row 1: missing required columns {missing!r}"
        )


def _cell(row: Mapping[str, str], column: str) -> str:
    value = row.get(column, "")
    return "" if value is None else value


def parse_decimal(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
    allow_blank: bool = False,
) -> Decimal | None:
    text = raw.strip()
    if text == "":
        if allow_blank:
            return None
        raise ParseError(
            f"{filename} row {row_number}: invalid Decimal {column}={raw!r}"
        )
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ParseError(
            f"{filename} row {row_number}: invalid Decimal {column}={raw!r}"
        ) from exc
    if not value.is_finite():
        raise ParseError(
            f"{filename} row {row_number}: invalid Decimal {column}={raw!r}"
        )
    return value


def parse_required_decimal(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> Decimal:
    value = parse_decimal(
        raw,
        filename=filename,
        row_number=row_number,
        column=column,
        allow_blank=False,
    )
    assert value is not None
    return value


def parse_int(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
    allow_blank: bool = False,
) -> int | None:
    text = raw.strip()
    if text == "":
        if allow_blank:
            return None
        raise ParseError(f"{filename} row {row_number}: invalid int {column}={raw!r}")
    try:
        value = int(text)
    except ValueError as exc:
        raise ParseError(
            f"{filename} row {row_number}: invalid int {column}={raw!r}"
        ) from exc
    return value


def parse_required_int(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> int:
    value = parse_int(
        raw,
        filename=filename,
        row_number=row_number,
        column=column,
        allow_blank=False,
    )
    assert value is not None
    return value


def parse_bool(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> bool:
    text = raw.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ParseError(f"{filename} row {row_number}: invalid bool {column}={raw!r}")


def parse_date(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
    allow_blank: bool = False,
) -> date | None:
    text = raw.strip()
    if text == "":
        if allow_blank:
            return None
        raise ParseError(f"{filename} row {row_number}: invalid date {column}={raw!r}")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ParseError(
            f"{filename} row {row_number}: invalid date {column}={raw!r}"
        ) from exc


def parse_required_date(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> date:
    value = parse_date(
        raw,
        filename=filename,
        row_number=row_number,
        column=column,
        allow_blank=False,
    )
    assert value is not None
    return value


def parse_datetime(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> datetime:
    text = raw.strip()
    if text == "":
        raise ParseError(
            f"{filename} row {row_number}: invalid datetime {column}={raw!r}"
        )
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ParseError(
            f"{filename} row {row_number}: invalid datetime {column}={raw!r}"
        ) from exc


def parse_enum(
    enum_cls: type[EnumT],
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> EnumT:
    text = raw.strip()
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise ParseError(
            f"{filename} row {row_number}: invalid {enum_cls.__name__} {text!r}"
        ) from exc


def parse_optional_text(raw: str) -> str | None:
    text = raw.strip()
    return None if text == "" else text


def parse_pipe_list(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if text == "":
        return ()
    return tuple(part for part in text.split("|") if part != "")


def parse_considered_payment_methods(
    raw: str,
    *,
    filename: str,
    row_number: int,
    column: str,
) -> tuple[ConsideredPaymentMethod, ...]:
    values = []
    for part in parse_pipe_list(raw):
        values.append(
            parse_enum(
                ConsideredPaymentMethod,
                part,
                filename=filename,
                row_number=row_number,
                column=column,
            )
        )
    return tuple(values)


def _read_rows(path: Path, required: Sequence[str]) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, required)
        for index, row in enumerate(reader, start=2):
            yield index, row


def load_profiles(path: Path) -> tuple[FinancialProfile, ...]:
    required = (
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    )
    filename = _filename(path)
    profiles: list[FinancialProfile] = []
    for row_number, row in _read_rows(path, required):
        profiles.append(
            FinancialProfile(
                user_id=_cell(row, "user_id").strip(),
                home_currency=parse_enum(
                    Currency,
                    _cell(row, "home_currency"),
                    filename=filename,
                    row_number=row_number,
                    column="home_currency",
                ),
                current_available_balance=parse_required_decimal(
                    _cell(row, "current_available_balance"),
                    filename=filename,
                    row_number=row_number,
                    column="current_available_balance",
                ),
                minimum_balance_to_keep=parse_required_decimal(
                    _cell(row, "minimum_balance_to_keep"),
                    filename=filename,
                    row_number=row_number,
                    column="minimum_balance_to_keep",
                ),
                financial_priorities=parse_pipe_list(_cell(row, "financial_priorities")),
                expense_categories_to_protect=parse_pipe_list(
                    _cell(row, "expense_categories_to_protect")
                ),
                expense_categories_user_is_willing_to_reduce=parse_pipe_list(
                    _cell(row, "expense_categories_user_is_willing_to_reduce")
                ),
                expense_categories_user_is_willing_to_stop=parse_pipe_list(
                    _cell(row, "expense_categories_user_is_willing_to_stop")
                ),
                payment_methods_user_will_consider=parse_considered_payment_methods(
                    _cell(row, "payment_methods_user_will_consider"),
                    filename=filename,
                    row_number=row_number,
                    column="payment_methods_user_will_consider",
                ),
                max_installment_months=parse_int(
                    _cell(row, "max_installment_months"),
                    filename=filename,
                    row_number=row_number,
                    column="max_installment_months",
                    allow_blank=True,
                ),
            )
        )
    return tuple(profiles)


def _load_finance_request(
    row: Mapping[str, str],
    *,
    filename: str,
    row_number: int,
) -> FinanceRequest:
    return FinanceRequest(
        request_id=_cell(row, "request_id").strip(),
        user_id=_cell(row, "user_id").strip(),
        request_date=parse_required_date(
            _cell(row, "request_date"),
            filename=filename,
            row_number=row_number,
            column="request_date",
        ),
        request_type=parse_enum(
            RequestType,
            _cell(row, "request_type"),
            filename=filename,
            row_number=row_number,
            column="request_type",
        ),
        requested_amount=parse_required_decimal(
            _cell(row, "requested_amount"),
            filename=filename,
            row_number=row_number,
            column="requested_amount",
        ),
        desired_completion_date=parse_required_date(
            _cell(row, "desired_completion_date"),
            filename=filename,
            row_number=row_number,
            column="desired_completion_date",
        ),
        allows_partial_payment=parse_bool(
            _cell(row, "allows_partial_payment"),
            filename=filename,
            row_number=row_number,
            column="allows_partial_payment",
        ),
        request_text=_cell(row, "request_text"),
    )


def load_evaluation_requests(path: Path) -> tuple[FinanceRequest, ...]:
    required = (
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    )
    filename = _filename(path)
    return tuple(
        _load_finance_request(row, filename=filename, row_number=row_number)
        for row_number, row in _read_rows(path, required)
    )


def load_sample_requests(path: Path) -> tuple[SampleFinanceRequest, ...]:
    required = (
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    )
    filename = _filename(path)
    samples: list[SampleFinanceRequest] = []
    for row_number, row in _read_rows(path, required):
        samples.append(
            SampleFinanceRequest(
                request=_load_finance_request(
                    row, filename=filename, row_number=row_number
                ),
                decision=SampleDecision(
                    amount_safe_to_pay=parse_required_decimal(
                        _cell(row, "amount_safe_to_pay"),
                        filename=filename,
                        row_number=row_number,
                        column="amount_safe_to_pay",
                    ),
                    affordability_status=parse_enum(
                        AffordabilityStatus,
                        _cell(row, "affordability_status"),
                        filename=filename,
                        row_number=row_number,
                        column="affordability_status",
                    ),
                    recommended_payment_method=parse_enum(
                        RecommendedPaymentMethod,
                        _cell(row, "recommended_payment_method"),
                        filename=filename,
                        row_number=row_number,
                        column="recommended_payment_method",
                    ),
                    payment_plan=_cell(row, "payment_plan"),
                    earliest_date_for_full_payment=parse_date(
                        _cell(row, "earliest_date_for_full_payment"),
                        filename=filename,
                        row_number=row_number,
                        column="earliest_date_for_full_payment",
                        allow_blank=True,
                    ),
                    spending_changes_needed=_cell(row, "spending_changes_needed"),
                    decision_explanation=_cell(row, "decision_explanation"),
                ),
            )
        )
    return tuple(samples)


def load_financial_events(path: Path) -> tuple[FinancialEvent, ...]:
    required = (
        "event_id",
        "user_id",
        "event_type",
        "description",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    )
    filename = _filename(path)
    events: list[FinancialEvent] = []
    for row_number, row in _read_rows(path, required):
        events.append(
            FinancialEvent(
                event_id=_cell(row, "event_id").strip(),
                user_id=_cell(row, "user_id").strip(),
                event_type=parse_enum(
                    EventType,
                    _cell(row, "event_type"),
                    filename=filename,
                    row_number=row_number,
                    column="event_type",
                ),
                description=_cell(row, "description"),
                category=_cell(row, "category").strip(),
                direction=parse_enum(
                    EventDirection,
                    _cell(row, "direction"),
                    filename=filename,
                    row_number=row_number,
                    column="direction",
                ),
                amount=parse_decimal(
                    _cell(row, "amount"),
                    filename=filename,
                    row_number=row_number,
                    column="amount",
                    allow_blank=True,
                ),
                currency=parse_enum(
                    Currency,
                    _cell(row, "currency"),
                    filename=filename,
                    row_number=row_number,
                    column="currency",
                ),
                event_date=parse_required_date(
                    _cell(row, "event_date"),
                    filename=filename,
                    row_number=row_number,
                    column="event_date",
                ),
                settlement_date=parse_date(
                    _cell(row, "settlement_date"),
                    filename=filename,
                    row_number=row_number,
                    column="settlement_date",
                    allow_blank=True,
                ),
                status=parse_enum(
                    EventStatus,
                    _cell(row, "status"),
                    filename=filename,
                    row_number=row_number,
                    column="status",
                ),
                linked_event_id=parse_optional_text(_cell(row, "linked_event_id")),
                flexibility=parse_enum(
                    Flexibility,
                    _cell(row, "flexibility"),
                    filename=filename,
                    row_number=row_number,
                    column="flexibility",
                ),
                minimum_allowed_amount=parse_decimal(
                    _cell(row, "minimum_allowed_amount"),
                    filename=filename,
                    row_number=row_number,
                    column="minimum_allowed_amount",
                    allow_blank=True,
                ),
            )
        )
    return tuple(events)


def load_exchange_rates(path: Path) -> tuple[ExchangeRate, ...]:
    required = ("rate_date", "from_currency", "to_currency", "rate")
    filename = _filename(path)
    rates: list[ExchangeRate] = []
    for row_number, row in _read_rows(path, required):
        rates.append(
            ExchangeRate(
                rate_date=parse_required_date(
                    _cell(row, "rate_date"),
                    filename=filename,
                    row_number=row_number,
                    column="rate_date",
                ),
                from_currency=parse_enum(
                    Currency,
                    _cell(row, "from_currency"),
                    filename=filename,
                    row_number=row_number,
                    column="from_currency",
                ),
                to_currency=parse_enum(
                    Currency,
                    _cell(row, "to_currency"),
                    filename=filename,
                    row_number=row_number,
                    column="to_currency",
                ),
                rate=parse_required_decimal(
                    _cell(row, "rate"),
                    filename=filename,
                    row_number=row_number,
                    column="rate",
                ),
            )
        )
    return tuple(rates)


def load_payment_options(path: Path) -> tuple[PaymentOption, ...]:
    required = (
        "payment_option_id",
        "request_id",
        "payment_method",
        "payment_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    )
    filename = _filename(path)
    options: list[PaymentOption] = []
    for row_number, row in _read_rows(path, required):
        options.append(
            PaymentOption(
                payment_option_id=_cell(row, "payment_option_id").strip(),
                request_id=_cell(row, "request_id").strip(),
                payment_method=parse_enum(
                    PaymentOptionMethod,
                    _cell(row, "payment_method"),
                    filename=filename,
                    row_number=row_number,
                    column="payment_method",
                ),
                payment_amount=parse_required_decimal(
                    _cell(row, "payment_amount"),
                    filename=filename,
                    row_number=row_number,
                    column="payment_amount",
                ),
                number_of_payments=parse_required_int(
                    _cell(row, "number_of_payments"),
                    filename=filename,
                    row_number=row_number,
                    column="number_of_payments",
                ),
                first_payment_date=parse_required_date(
                    _cell(row, "first_payment_date"),
                    filename=filename,
                    row_number=row_number,
                    column="first_payment_date",
                ),
                payment_frequency_days=parse_int(
                    _cell(row, "payment_frequency_days"),
                    filename=filename,
                    row_number=row_number,
                    column="payment_frequency_days",
                    allow_blank=True,
                ),
                financing_fee=parse_required_decimal(
                    _cell(row, "financing_fee"),
                    filename=filename,
                    row_number=row_number,
                    column="financing_fee",
                ),
                total_payable_amount=parse_required_decimal(
                    _cell(row, "total_payable_amount"),
                    filename=filename,
                    row_number=row_number,
                    column="total_payable_amount",
                ),
            )
        )
    return tuple(options)


def load_messages(path: Path) -> tuple[Message, ...]:
    required = (
        "message_id",
        "user_id",
        "request_id",
        "related_event_id",
        "sent_at",
        "source_type",
        "message_text",
    )
    filename = _filename(path)
    messages: list[Message] = []
    for row_number, row in _read_rows(path, required):
        messages.append(
            Message(
                message_id=_cell(row, "message_id").strip(),
                user_id=_cell(row, "user_id").strip(),
                request_id=parse_optional_text(_cell(row, "request_id")),
                related_event_id=parse_optional_text(_cell(row, "related_event_id")),
                sent_at=parse_datetime(
                    _cell(row, "sent_at"),
                    filename=filename,
                    row_number=row_number,
                    column="sent_at",
                ),
                source_type=parse_enum(
                    MessageSourceType,
                    _cell(row, "source_type"),
                    filename=filename,
                    row_number=row_number,
                    column="source_type",
                ),
                message_text=_cell(row, "message_text"),
            )
        )
    return tuple(messages)


def load_images(path: Path, images_dir: Path) -> tuple[ImageReference, ...]:
    required = ("image_id", "user_id", "request_id", "related_event_id")
    images: list[ImageReference] = []
    for _row_number, row in _read_rows(path, required):
        image_id = _cell(row, "image_id").strip()
        images.append(
            ImageReference(
                image_id=image_id,
                user_id=_cell(row, "user_id").strip(),
                request_id=_cell(row, "request_id").strip(),
                related_event_id=_cell(row, "related_event_id").strip(),
                path=images_dir / f"{image_id}.png",
            )
        )
    return tuple(images)


def load_output_template(path: Path) -> tuple[OutputTemplateRow, ...]:
    required = (
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    )
    filename = _filename(path)
    rows: list[OutputTemplateRow] = []
    for row_number, row in _read_rows(path, required):
        status_raw = _cell(row, "affordability_status").strip()
        method_raw = _cell(row, "recommended_payment_method").strip()
        rows.append(
            OutputTemplateRow(
                request_id=_cell(row, "request_id").strip(),
                amount_safe_to_pay=parse_decimal(
                    _cell(row, "amount_safe_to_pay"),
                    filename=filename,
                    row_number=row_number,
                    column="amount_safe_to_pay",
                    allow_blank=True,
                ),
                affordability_status=(
                    parse_enum(
                        AffordabilityStatus,
                        status_raw,
                        filename=filename,
                        row_number=row_number,
                        column="affordability_status",
                    )
                    if status_raw
                    else None
                ),
                recommended_payment_method=(
                    parse_enum(
                        RecommendedPaymentMethod,
                        method_raw,
                        filename=filename,
                        row_number=row_number,
                        column="recommended_payment_method",
                    )
                    if method_raw
                    else None
                ),
                payment_plan=parse_optional_text(_cell(row, "payment_plan")),
                earliest_date_for_full_payment=parse_date(
                    _cell(row, "earliest_date_for_full_payment"),
                    filename=filename,
                    row_number=row_number,
                    column="earliest_date_for_full_payment",
                    allow_blank=True,
                ),
                spending_changes_needed=parse_optional_text(
                    _cell(row, "spending_changes_needed")
                ),
                decision_explanation=parse_optional_text(
                    _cell(row, "decision_explanation")
                ),
            )
        )
    return tuple(rows)


def assemble_dataset(
    *,
    profiles: tuple[FinancialProfile, ...],
    evaluation_requests: tuple[FinanceRequest, ...],
    sample_requests: tuple[SampleFinanceRequest, ...],
    events: tuple[FinancialEvent, ...],
    exchange_rates: tuple[ExchangeRate, ...],
    payment_options: tuple[PaymentOption, ...],
    messages: tuple[Message, ...],
    images: tuple[ImageReference, ...],
    output_template_rows: tuple[OutputTemplateRow, ...],
) -> Dataset:
    return Dataset(
        profiles=profiles,
        evaluation_requests=evaluation_requests,
        sample_requests=sample_requests,
        events=events,
        exchange_rates=exchange_rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
        output_template_rows=output_template_rows,
        profiles_by_user_id={profile.user_id: profile for profile in profiles},
        evaluation_requests_by_id={
            request.request_id: request for request in evaluation_requests
        },
        sample_requests_by_id={
            sample.request_id: sample for sample in sample_requests
        },
        events_by_id={event.event_id: event for event in events},
        exchange_rates_by_key={rate.key: rate for rate in exchange_rates},
        payment_options_by_id={
            option.payment_option_id: option for option in payment_options
        },
        messages_by_id={message.message_id: message for message in messages},
        images_by_id={image.image_id: image for image in images},
        output_template_by_request_id={
            row.request_id: row for row in output_template_rows
        },
    )


def load_dataset(paths: DatasetPaths | None = None) -> Dataset:
    dataset_paths = paths or DatasetPaths.default()
    return assemble_dataset(
        profiles=load_profiles(dataset_paths.financial_profiles),
        evaluation_requests=load_evaluation_requests(dataset_paths.evaluation_requests),
        sample_requests=load_sample_requests(dataset_paths.sample_requests),
        events=load_financial_events(dataset_paths.financial_events),
        exchange_rates=load_exchange_rates(dataset_paths.exchange_rates),
        payment_options=load_payment_options(dataset_paths.payment_options),
        messages=load_messages(dataset_paths.messages),
        images=load_images(dataset_paths.images, dataset_paths.images_dir),
        output_template_rows=load_output_template(dataset_paths.output_template),
    )
