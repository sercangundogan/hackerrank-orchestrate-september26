from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from data.loader import (
    ParseError,
    load_evaluation_requests,
    load_financial_events,
    load_messages,
    load_profiles,
    parse_bool,
    parse_date,
    parse_datetime,
    parse_decimal,
    parse_enum,
    parse_pipe_list,
)
from data.models import (
    Currency,
    Dataset,
    EventStatus,
    EventType,
    Flexibility,
    RequestType,
)
from tests.conftest import write_csv


def test_all_csv_files_load(dataset: Dataset) -> None:
    assert len(dataset.profiles) == 275
    assert len(dataset.evaluation_requests) == 250
    assert len(dataset.sample_requests) == 25
    assert len(dataset.events) == 25342
    assert len(dataset.exchange_rates) == 134
    assert len(dataset.payment_options) == 790
    assert len(dataset.messages) == 215
    assert len(dataset.images) == 16
    assert len(dataset.output_template_rows) == 250


def test_money_values_are_decimal(dataset: Dataset) -> None:
    profile = dataset.profiles_by_user_id["user_01"]
    assert isinstance(profile.current_available_balance, Decimal)
    assert profile.current_available_balance == Decimal("58481.1")
    assert isinstance(profile.minimum_balance_to_keep, Decimal)
    assert profile.minimum_balance_to_keep == Decimal("18000")
    request = dataset.evaluation_requests_by_id["request_26"]
    assert isinstance(request.requested_amount, Decimal)
    event = dataset.events_by_id["event_01"]
    assert isinstance(event.amount, Decimal)
    assert event.amount == Decimal("5148")
    rate = dataset.exchange_rates[0]
    assert isinstance(rate.rate, Decimal)


def test_dates_and_datetimes_are_parsed_objects(dataset: Dataset) -> None:
    request = dataset.evaluation_requests_by_id["request_26"]
    assert request.request_date == date(2025, 8, 3)
    assert request.desired_completion_date == date(2025, 10, 7)
    event = dataset.events_by_id["event_01"]
    assert event.event_date == date(2023, 10, 2)
    assert event.settlement_date == date(2023, 10, 2)
    message = dataset.messages_by_id["message_01"]
    assert message.sent_at == datetime(2025, 7, 29, 9, 30, tzinfo=timezone.utc)


def test_pipe_lists_and_nullable_profile_fields(dataset: Dataset) -> None:
    profile = dataset.profiles_by_user_id["user_01"]
    assert profile.financial_priorities == ("education", "debt_repayment")
    assert profile.expense_categories_to_protect == (
        "rent",
        "education",
        "groceries",
        "debt_repayment",
    )
    assert profile.max_installment_months is None
    user_two = dataset.profiles_by_user_id["user_02"]
    assert user_two.max_installment_months == 7
    user_nine = dataset.profiles_by_user_id["user_09"]
    assert user_nine.expense_categories_user_is_willing_to_reduce == ()
    assert user_nine.expense_categories_user_is_willing_to_stop == ()


def test_boolean_parsing_on_requests(dataset: Dataset) -> None:
    assert dataset.evaluation_requests_by_id["request_26"].allows_partial_payment is False
    assert dataset.evaluation_requests_by_id["request_27"].allows_partial_payment is True


def test_blank_event_amount_is_none_not_zero(dataset: Dataset) -> None:
    event = dataset.events_by_id["event_253"]
    assert event.amount is None
    assert event.amount != Decimal("0")


def test_normal_event_amount_is_decimal(dataset: Dataset) -> None:
    event = dataset.events_by_id["event_01"]
    assert event.amount == Decimal("5148")
    assert event.event_type is EventType.EXPENSE
    assert event.flexibility is Flexibility.FIXED


def test_unrealized_settlement_date_is_none(dataset: Dataset) -> None:
    event = dataset.events_by_id["event_1856"]
    assert event.status is EventStatus.UNREALIZED
    assert event.settlement_date is None


def test_linked_event_id_parsing(dataset: Dataset) -> None:
    assert dataset.events_by_id["event_01"].linked_event_id is None
    refund = dataset.events_by_id["event_99"]
    assert refund.linked_event_id == "event_98"


def test_minimum_allowed_amount_parsing(dataset: Dataset) -> None:
    reducible = dataset.events_by_id["event_85"]
    assert reducible.flexibility is Flexibility.REDUCIBLE
    assert reducible.minimum_allowed_amount == Decimal("489.5")
    fixed = dataset.events_by_id["event_01"]
    assert fixed.minimum_allowed_amount is None


def test_sample_output_fields_remain_accessible(dataset: Dataset) -> None:
    sample = dataset.sample_requests_by_id["request_01"]
    assert sample.request.request_type is RequestType.PURCHASE
    assert sample.decision.amount_safe_to_pay == Decimal("25256")
    unaffordable = dataset.sample_requests_by_id["request_05"]
    assert unaffordable.decision.earliest_date_for_full_payment is None


def test_parse_decimal_helpers() -> None:
    assert parse_decimal(
        "12.50", filename="x.csv", row_number=2, column="amount"
    ) == Decimal("12.50")
    assert (
        parse_decimal(
            "", filename="x.csv", row_number=2, column="amount", allow_blank=True
        )
        is None
    )
    with pytest.raises(ParseError, match="x.csv row 17: invalid Decimal requested_amount='abc'"):
        parse_decimal("abc", filename="x.csv", row_number=17, column="requested_amount")
    with pytest.raises(ParseError, match="invalid Decimal"):
        parse_decimal("", filename="x.csv", row_number=2, column="amount")


def test_parse_bool_rejects_silent_coercion() -> None:
    assert parse_bool("true", filename="x.csv", row_number=2, column="flag") is True
    assert parse_bool("false", filename="x.csv", row_number=2, column="flag") is False
    with pytest.raises(ParseError, match="invalid bool"):
        parse_bool("yes", filename="x.csv", row_number=2, column="flag")


def test_parse_date_and_datetime() -> None:
    assert parse_date(
        "2024-03-03", filename="x.csv", row_number=2, column="request_date"
    ) == date(2024, 3, 3)
    assert (
        parse_date(
            "",
            filename="x.csv",
            row_number=2,
            column="settlement_date",
            allow_blank=True,
        )
        is None
    )
    with pytest.raises(ParseError, match="invalid date"):
        parse_date("03-03-2024", filename="x.csv", row_number=2, column="request_date")
    parsed = parse_datetime(
        "2025-07-29T09:30:00Z", filename="x.csv", row_number=2, column="sent_at"
    )
    assert parsed == datetime(2025, 7, 29, 9, 30, tzinfo=timezone.utc)


def test_parse_pipe_list() -> None:
    assert parse_pipe_list("education|debt_repayment") == ("education", "debt_repayment")
    assert parse_pipe_list("") == ()
    assert parse_pipe_list("  dining  ") == ("dining",)


def test_invalid_enum_from_csv(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "financial_events.csv",
        [
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
        ],
        [
            {
                "event_id": "event_x",
                "user_id": "user_01",
                "event_type": "expense",
                "description": "test",
                "category": "rent",
                "direction": "debit",
                "amount": "10",
                "currency": "ZAR",
                "event_date": "2024-01-01",
                "settlement_date": "2024-01-01",
                "status": "xyz",
                "linked_event_id": "",
                "flexibility": "fixed",
                "minimum_allowed_amount": "",
            }
        ],
    )
    with pytest.raises(ParseError, match="financial_events.csv row 2: invalid EventStatus 'xyz'"):
        load_financial_events(path)


def test_invalid_decimal_from_requests_csv(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "requests.csv",
        [
            "request_id",
            "user_id",
            "request_date",
            "request_type",
            "requested_amount",
            "desired_completion_date",
            "allows_partial_payment",
            "request_text",
        ],
        [
            {
                "request_id": "request_x",
                "user_id": "user_01",
                "request_date": "2024-01-01",
                "request_type": "purchase",
                "requested_amount": "abc",
                "desired_completion_date": "2024-02-01",
                "allows_partial_payment": "false",
                "request_text": "test",
            }
        ],
    )
    with pytest.raises(
        ParseError, match="requests.csv row 2: invalid Decimal requested_amount='abc'"
    ):
        load_evaluation_requests(path)


def test_missing_required_column(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "financial_profiles.csv",
        ["user_id", "home_currency"],
        [{"user_id": "user_01", "home_currency": "ZAR"}],
    )
    with pytest.raises(ParseError, match="missing required columns"):
        load_profiles(path)


def test_parse_enum_direct() -> None:
    assert parse_enum(
        Currency, "INR", filename="x.csv", row_number=2, column="currency"
    ) is Currency.INR
    with pytest.raises(ParseError, match="invalid Currency 'GBP'"):
        parse_enum(Currency, "GBP", filename="x.csv", row_number=2, column="currency")


def test_message_optional_ids(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "messages.csv",
        [
            "message_id",
            "user_id",
            "request_id",
            "related_event_id",
            "sent_at",
            "source_type",
            "message_text",
        ],
        [
            {
                "message_id": "message_x",
                "user_id": "user_01",
                "request_id": "",
                "related_event_id": "",
                "sent_at": "2025-07-29T09:30:00Z",
                "source_type": "employer",
                "message_text": "hello",
            }
        ],
    )
    messages = load_messages(path)
    assert messages[0].request_id is None
    assert messages[0].related_event_id is None
