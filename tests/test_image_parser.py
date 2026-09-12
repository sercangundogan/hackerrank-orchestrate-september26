from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ai.client import InvalidModelOutputError, ModelClient, ModelResponse
from data.models import (
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinanceRequest,
    Flexibility,
    ImageReference,
    RequestType,
)
from evidence.image_parser import _validate_image_payload, parse_image
from evidence.models import EvidenceConfidence
from tests.factories import make_event


class FakeClient(ModelClient):
    def __init__(self, payload: dict, *, fail: bool = False) -> None:
        super().__init__(api_key="test")
        self.payload = payload
        self.fail = fail
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete_json(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail:
            raise InvalidModelOutputError("bad json")
        return ModelResponse(
            payload=self.payload,
            provider="fake",
            model="fake-vision",
            input_tokens=10,
            output_tokens=5,
            prompt_version="v1",
        )


def _request() -> FinanceRequest:
    return FinanceRequest(
        request_id="request_x",
        user_id="user_x",
        request_date=date(2026, 1, 1),
        request_type=RequestType.PURCHASE,
        requested_amount=Decimal("1"),
        desired_completion_date=date(2026, 1, 10),
        allows_partial_payment=False,
        request_text="test",
    )


def test_payslip_uses_net_pay_not_gross() -> None:
    event = make_event(
        event_id="event_253",
        event_type=EventType.INCOME,
        category="salary",
        direction=EventDirection.CREDIT,
        description="August 2019 net salary",
        currency=Currency.IDR,
        amount=None,
        status=EventStatus.SETTLED,
    )
    extraction = _validate_image_payload(
        {
            "amount": "4365000",
            "currency": "IDR",
            "selected_label": "Net payable",
            "confidence": "high",
            "rationale": "payslip net pay, not gross earnings",
            "related_event_id": "event_253",
        },
        event,
    )
    assert extraction.amount == Decimal("4365000")
    assert extraction.selected_label == "Net payable"


def test_invoice_uses_balance_due() -> None:
    event = make_event(
        event_id="event_1442",
        category="rent",
        description="Outstanding rent balance",
        currency=Currency.INR,
        amount=None,
        status=EventStatus.SCHEDULED,
    )
    extraction = _validate_image_payload(
        {
            "amount": "100000",
            "currency": "INR",
            "selected_label": "Balance due",
            "confidence": "high",
            "rationale": "event is outstanding balance, not total billed",
            "related_event_id": "event_1442",
        },
        event,
    )
    assert extraction.amount == Decimal("100000")


def test_receipt_grand_total() -> None:
    event = make_event(
        event_id="event_g",
        category="groceries",
        description="Delivered grocery order",
        currency=Currency.INR,
        amount=None,
    )
    extraction = _validate_image_payload(
        {
            "amount": "1840.50",
            "currency": "INR",
            "selected_label": "Grand total",
            "confidence": "medium",
            "rationale": "grocery receipt total",
            "related_event_id": "event_g",
        },
        event,
    )
    assert extraction.amount == Decimal("1840.50")


def test_cash_paid_is_not_taxi_fare() -> None:
    event = make_event(
        event_id="event_7307",
        category="transport",
        description="Taxi fare",
        currency=Currency.USD,
        amount=None,
    )
    extraction = _validate_image_payload(
        {
            "amount": "33.50",
            "currency": "USD",
            "selected_label": "Total",
            "confidence": "high",
            "rationale": "fare total, not cash tendered 40.00",
            "related_event_id": "event_7307",
        },
        event,
    )
    assert extraction.amount == Decimal("33.50")


def test_malformed_vlm_output_rejected() -> None:
    event = make_event(event_id="event_x", amount=None)
    with pytest.raises(ValueError):
        _validate_image_payload(
            {
                "amount": "10",
                "currency": "INR",
                "confidence": "high",
                "rationale": "invented id",
                "related_event_id": "event_other",
            },
            event,
        )
    with pytest.raises(ValueError):
        _validate_image_payload(
            {
                "amount": "-5",
                "currency": "INR",
                "confidence": "high",
                "rationale": "negative",
                "related_event_id": "event_x",
            },
            event,
        )


def test_parse_image_records_failure_without_zero(tmp_path: Path) -> None:
    path = tmp_path / "image_x.png"
    path.write_bytes(b"png")
    image = ImageReference(
        image_id="image_x",
        user_id="user_x",
        request_id="request_x",
        related_event_id="event_x",
        path=path,
    )
    event = make_event(event_id="event_x", amount=None)
    result = parse_image(image, event, _request(), client=FakeClient({}, fail=True))
    assert result.amount is None
    assert result.unresolved_reason
