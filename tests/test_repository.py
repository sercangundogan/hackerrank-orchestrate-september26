from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from data.models import Currency
from data.repository import DatasetRepository


def test_profile_lookup(repository: DatasetRepository) -> None:
    profile = repository.profile_for_user("user_01")
    assert profile.home_currency is Currency.ZAR
    assert profile.current_available_balance == Decimal("58481.1")
    with pytest.raises(LookupError, match="unknown user_id 'user_missing'"):
        repository.profile_for_user("user_missing")


def test_evaluation_and_sample_request_lookup(repository: DatasetRepository) -> None:
    evaluation = repository.request_by_id("request_26")
    assert evaluation.user_id == "user_26"
    sample = repository.sample_request_by_id("request_01")
    assert sample.user_id == "user_01"
    with pytest.raises(LookupError, match="unknown evaluation request_id 'request_01'"):
        repository.request_by_id("request_01")
    with pytest.raises(LookupError, match="unknown sample request_id 'request_26'"):
        repository.sample_request_by_id("request_26")


def test_event_lookup_and_events_for_user(repository: DatasetRepository) -> None:
    event = repository.event_by_id("event_01")
    assert event.user_id == "user_01"
    user_events = repository.events_for_user("user_01")
    assert user_events
    assert all(item.user_id == "user_01" for item in user_events)
    assert repository.event_by_id("event_01") in user_events
    assert repository.events_for_user("user_missing") == ()
    with pytest.raises(LookupError, match="unknown event_id"):
        repository.event_by_id("event_missing")


def test_payment_options_lookup(repository: DatasetRepository) -> None:
    options = repository.payment_options_for_request("request_01")
    assert len(options) == 4
    assert {option.payment_method.value for option in options} == {
        "full_payment",
        "installments",
    }
    assert repository.payment_options_for_request("request_missing") == ()


def test_message_lookups(repository: DatasetRepository) -> None:
    user_messages = repository.messages_for_user("user_02")
    assert len(user_messages) == 1
    assert user_messages[0].message_id == "message_01"
    assert repository.messages_for_user("user_missing") == ()
    request_messages = repository.messages_for_request("request_03")
    assert request_messages[0].message_id == "message_02"
    event_messages = repository.messages_for_event("event_1785")
    assert event_messages[0].message_id == "message_14"
    assert repository.messages_for_event("event_01") == ()


def test_image_lookups(repository: DatasetRepository) -> None:
    images = repository.images_for_user("user_03")
    assert len(images) == 1
    assert images[0].image_id == "image_01"
    request_images = repository.images_for_request("request_03")
    assert request_images[0].related_event_id == "event_253"
    image = repository.image_for_event("event_253")
    assert image is not None
    assert image.path.name == "image_01.png"
    assert image.path.is_file()
    assert repository.image_for_event("event_01") is None
    assert repository.images_for_user("user_missing") == ()


def test_exact_exchange_rate_lookup(repository: DatasetRepository) -> None:
    rate = repository.exchange_rate(date(2023, 10, 15), Currency.USD, Currency.EUR)
    assert rate.rate == Decimal("0.92")
    assert rate.from_currency is Currency.USD
    assert rate.to_currency is Currency.EUR


def test_exchange_rate_does_not_invert_or_guess(repository: DatasetRepository) -> None:
    with pytest.raises(LookupError, match="no exact exchange rate"):
        repository.exchange_rate(date(2023, 10, 15), Currency.EUR, Currency.USD)
    with pytest.raises(LookupError, match="no exact exchange rate"):
        repository.exchange_rate(date(2023, 10, 14), Currency.USD, Currency.EUR)
    with pytest.raises(LookupError, match="no exact exchange rate"):
        repository.exchange_rate(date(2023, 10, 15), Currency.IDR, Currency.USD)
