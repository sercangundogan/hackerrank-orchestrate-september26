"""Indexed query layer over a loaded Dataset.

Single-entity lookups raise LookupError. Collection queries return empty
tuples when nothing matches. Exchange-rate lookup is exact only: no inverse
conversion and no neighboring-date fallback.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from data.models import (
    Currency,
    Dataset,
    ExchangeRate,
    FinanceRequest,
    FinancialEvent,
    FinancialProfile,
    ImageReference,
    Message,
    PaymentOption,
    SampleFinanceRequest,
)


class DatasetRepository:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        events_by_user: dict[str, list[FinancialEvent]] = defaultdict(list)
        for event in dataset.events:
            events_by_user[event.user_id].append(event)
        self._events_by_user = {
            user_id: tuple(events) for user_id, events in events_by_user.items()
        }

        options_by_request: dict[str, list[PaymentOption]] = defaultdict(list)
        for option in dataset.payment_options:
            options_by_request[option.request_id].append(option)
        self._options_by_request = {
            request_id: tuple(options)
            for request_id, options in options_by_request.items()
        }

        messages_by_user: dict[str, list[Message]] = defaultdict(list)
        messages_by_request: dict[str, list[Message]] = defaultdict(list)
        messages_by_event: dict[str, list[Message]] = defaultdict(list)
        for message in dataset.messages:
            messages_by_user[message.user_id].append(message)
            if message.request_id is not None:
                messages_by_request[message.request_id].append(message)
            if message.related_event_id is not None:
                messages_by_event[message.related_event_id].append(message)
        self._messages_by_user = {
            user_id: tuple(items) for user_id, items in messages_by_user.items()
        }
        self._messages_by_request = {
            request_id: tuple(items)
            for request_id, items in messages_by_request.items()
        }
        self._messages_by_event = {
            event_id: tuple(items) for event_id, items in messages_by_event.items()
        }

        images_by_user: dict[str, list[ImageReference]] = defaultdict(list)
        images_by_request: dict[str, list[ImageReference]] = defaultdict(list)
        images_by_event: dict[str, ImageReference] = {}
        for image in dataset.images:
            images_by_user[image.user_id].append(image)
            images_by_request[image.request_id].append(image)
            images_by_event[image.related_event_id] = image
        self._images_by_user = {
            user_id: tuple(items) for user_id, items in images_by_user.items()
        }
        self._images_by_request = {
            request_id: tuple(items)
            for request_id, items in images_by_request.items()
        }
        self._images_by_event = images_by_event

    def profile_for_user(self, user_id: str) -> FinancialProfile:
        try:
            return self.dataset.profiles_by_user_id[user_id]
        except KeyError as exc:
            raise LookupError(f"unknown user_id {user_id!r}") from exc

    def request_by_id(self, request_id: str) -> FinanceRequest:
        try:
            return self.dataset.evaluation_requests_by_id[request_id]
        except KeyError as exc:
            raise LookupError(f"unknown evaluation request_id {request_id!r}") from exc

    def sample_request_by_id(self, request_id: str) -> SampleFinanceRequest:
        try:
            return self.dataset.sample_requests_by_id[request_id]
        except KeyError as exc:
            raise LookupError(f"unknown sample request_id {request_id!r}") from exc

    def events_for_user(self, user_id: str) -> tuple[FinancialEvent, ...]:
        return self._events_by_user.get(user_id, ())

    def event_by_id(self, event_id: str) -> FinancialEvent:
        try:
            return self.dataset.events_by_id[event_id]
        except KeyError as exc:
            raise LookupError(f"unknown event_id {event_id!r}") from exc

    def payment_options_for_request(self, request_id: str) -> tuple[PaymentOption, ...]:
        return self._options_by_request.get(request_id, ())

    def messages_for_user(self, user_id: str) -> tuple[Message, ...]:
        return self._messages_by_user.get(user_id, ())

    def messages_for_request(self, request_id: str) -> tuple[Message, ...]:
        return self._messages_by_request.get(request_id, ())

    def messages_for_event(self, event_id: str) -> tuple[Message, ...]:
        return self._messages_by_event.get(event_id, ())

    def images_for_user(self, user_id: str) -> tuple[ImageReference, ...]:
        return self._images_by_user.get(user_id, ())

    def images_for_request(self, request_id: str) -> tuple[ImageReference, ...]:
        return self._images_by_request.get(request_id, ())

    def image_for_event(self, event_id: str) -> ImageReference | None:
        return self._images_by_event.get(event_id)

    def exchange_rate(
        self,
        rate_date: date,
        from_currency: Currency,
        to_currency: Currency,
    ) -> ExchangeRate:
        key = (rate_date, from_currency, to_currency)
        try:
            return self.dataset.exchange_rates_by_key[key]
        except KeyError as exc:
            raise LookupError(
                "no exact exchange rate for "
                f"date={rate_date.isoformat()} "
                f"from={from_currency.value} to={to_currency.value}"
            ) from exc
