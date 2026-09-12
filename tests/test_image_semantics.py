from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from ai.client import ModelClient, ModelResponse
from data.models import (
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    FinanceRequest,
    ImageReference,
    RequestType,
)
from evidence.models import EvidenceConfidence, ExtractionMethod, ImageExtraction, ValidationStatus
from evidence.review import review_image_extraction
from evidence.semantic import validate_image_semantics
from tests.factories import make_event


class FakeClient(ModelClient):
    def __init__(self, payload: dict) -> None:
        super().__init__(api_key="test")
        self.payload = payload
        self.calls = 0
        self.purposes: list[str] = []

    def available(self) -> bool:
        return True

    def complete_json(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.purposes.append(str(kwargs.get("purpose")))
        return ModelResponse(
            payload=self.payload,
            provider="fake",
            model="fake-vision",
            input_tokens=8,
            output_tokens=4,
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


def _image(tmp_path: Path) -> ImageReference:
    path = tmp_path / "image_x.png"
    path.write_bytes(b"png")
    return ImageReference(
        image_id="image_x",
        user_id="user_x",
        request_id="request_x",
        related_event_id="event_x",
        path=path,
    )


def _extraction(
    *,
    event_id: str,
    amount: str,
    field: str,
    confidence: EvidenceConfidence = EvidenceConfidence.HIGH,
    currency: Currency = Currency.INR,
) -> ImageExtraction:
    return ImageExtraction(
        image_id="image_x",
        related_event_id=event_id,
        amount=Decimal(amount),
        currency=currency,
        confidence=confidence,
        extraction_method=ExtractionMethod.VLM,
        rationale="test extraction",
        selected_label=field,
    )


def test_salary_net_pay_is_valid() -> None:
    event = make_event(
        event_id="event_salary",
        event_type=EventType.INCOME,
        category="salary",
        direction=EventDirection.CREDIT,
        description="August net salary",
        amount=None,
        status=EventStatus.SETTLED,
        currency=Currency.IDR,
    )
    result = validate_image_semantics(
        event,
        _extraction(event_id="event_salary", amount="4365000", field="net_pay", currency=Currency.IDR),
    )
    assert result.status is ValidationStatus.VALID
    assert result.semantic_consistency is True


def test_grocery_net_pay_needs_review() -> None:
    event = make_event(
        event_id="event_grocery",
        category="groceries",
        description="Bulk groceries",
        amount=None,
        currency=Currency.INR,
    )
    result = validate_image_semantics(
        event,
        _extraction(event_id="event_grocery", amount="41272.0", field="net_pay"),
    )
    assert result.status is ValidationStatus.NEEDS_REVIEW
    assert result.semantic_consistency is False


def test_grocery_invoice_net_amount_is_valid() -> None:
    event = make_event(
        event_id="event_grocery",
        category="groceries",
        description="Bulk groceries",
        amount=None,
        currency=Currency.INR,
    )
    result = validate_image_semantics(
        event,
        _extraction(event_id="event_grocery", amount="41272.0", field="net_amount"),
    )
    assert result.status is ValidationStatus.VALID


def test_grocery_grand_total_is_valid() -> None:
    event = make_event(
        event_id="event_grocery",
        category="groceries",
        description="Delivered grocery order",
        amount=None,
        currency=Currency.INR,
    )
    result = validate_image_semantics(
        event,
        _extraction(event_id="event_grocery", amount="2854.0", field="grand_total"),
    )
    assert result.status is ValidationStatus.VALID


def test_taxi_cash_tendered_is_invalid() -> None:
    event = make_event(
        event_id="event_taxi",
        category="transport",
        description="Taxi fare",
        amount=None,
        currency=Currency.USD,
    )
    result = validate_image_semantics(
        event,
        _extraction(
            event_id="event_taxi",
            amount="40.00",
            field="cash_tendered",
            currency=Currency.USD,
        ),
    )
    assert result.status is ValidationStatus.INVALID
    assert result.semantic_consistency is False


def test_taxi_fare_total_is_valid() -> None:
    event = make_event(
        event_id="event_taxi",
        category="transport",
        description="Taxi fare",
        amount=None,
        currency=Currency.USD,
    )
    result = validate_image_semantics(
        event,
        _extraction(
            event_id="event_taxi",
            amount="33.50",
            field="fare_total",
            currency=Currency.USD,
        ),
    )
    assert result.status is ValidationStatus.VALID


def test_bill_balance_due_is_valid() -> None:
    event = make_event(
        event_id="event_rent",
        category="rent",
        description="Outstanding rent balance",
        amount=None,
        currency=Currency.INR,
        status=EventStatus.SCHEDULED,
    )
    result = validate_image_semantics(
        event,
        _extraction(event_id="event_rent", amount="100000.00", field="balance_due"),
    )
    assert result.status is ValidationStatus.VALID


def test_high_confidence_does_not_override_semantic_mismatch() -> None:
    event = make_event(
        event_id="event_grocery",
        category="groceries",
        description="Bulk groceries",
        amount=None,
        currency=Currency.INR,
    )
    result = validate_image_semantics(
        event,
        _extraction(
            event_id="event_grocery",
            amount="41272.0",
            field="net_pay",
            confidence=EvidenceConfidence.HIGH,
        ),
    )
    assert result.status is ValidationStatus.NEEDS_REVIEW
    assert result.extraction_confidence is EvidenceConfidence.HIGH
    assert result.semantic_consistency is False


def test_verification_invoice_net_amount_clarifies_grocery_net_pay(tmp_path: Path) -> None:
    event = make_event(
        event_id="event_x",
        category="groceries",
        description="Bulk groceries",
        amount=None,
        currency=Currency.INR,
    )
    original = _extraction(event_id="event_x", amount="41272.0", field="net_pay")
    client = FakeClient(
        {
            "event_id": "event_x",
            "amount": "41272.0",
            "currency": "INR",
            "semantic_field_selected": "Net Amount",
            "supporting_text": "Net Amount 41,272.00",
            "first_amount_correct": True,
            "confidence": "high",
            "rationale": "invoice net amount equals first extraction",
        }
    )
    reviewed = review_image_extraction(
        _image(tmp_path),
        event,
        _request(),
        original,
        client=client,
    )
    assert original.selected_label == "net_pay"
    assert reviewed.accepted is True
    assert reviewed.final_field == "net_amount"
    assert reviewed.final_amount == Decimal("41272.0")


def test_successful_verification_clarifies_semantic_field(tmp_path: Path) -> None:
    event = make_event(
        event_id="event_x",
        category="groceries",
        description="Bulk groceries",
        amount=None,
        currency=Currency.INR,
    )
    original = _extraction(event_id="event_x", amount="41272.0", field="net_pay")
    client = FakeClient(
        {
            "event_id": "event_x",
            "amount": "41272.0",
            "currency": "INR",
            "semantic_field_selected": "grand_total",
            "supporting_text": "Grand Total 41,272.00",
            "first_amount_correct": True,
            "confidence": "high",
            "rationale": "receipt grand total; first amount is correct",
        }
    )
    reviewed = review_image_extraction(
        _image(tmp_path),
        event,
        _request(),
        original,
        client=client,
    )
    assert client.calls == 1
    assert client.purposes == ["image_amount_verify"]
    assert original.selected_label == "net_pay"
    assert reviewed.accepted is True
    assert reviewed.final_field == "grand_total"
    assert reviewed.final_amount == Decimal("41272.0")
    assert reviewed.validation.status is ValidationStatus.NEEDS_REVIEW


def test_conflicting_verification_remains_unresolved(tmp_path: Path) -> None:
    event = make_event(
        event_id="event_x",
        category="groceries",
        description="Bulk groceries",
        amount=None,
        currency=Currency.INR,
    )
    original = _extraction(event_id="event_x", amount="41272.0", field="net_pay")
    client = FakeClient(
        {
            "event_id": "event_x",
            "amount": "39999.00",
            "currency": "INR",
            "semantic_field_selected": "grand_total",
            "supporting_text": "another total",
            "first_amount_correct": False,
            "confidence": "medium",
            "rationale": "different visible total",
        }
    )
    reviewed = review_image_extraction(
        _image(tmp_path),
        event,
        _request(),
        original,
        client=client,
    )
    assert reviewed.accepted is False
    assert reviewed.final_amount is None
    assert reviewed.candidates == ("net_pay:41272.0", "grand_total:39999.00")
    assert "disagree" in (reviewed.unresolved_reason or "")


def test_valid_extraction_does_not_call_verification(tmp_path: Path) -> None:
    event = make_event(
        event_id="event_x",
        category="groceries",
        description="Delivered grocery order",
        amount=None,
        currency=Currency.INR,
    )
    client = FakeClient({})
    reviewed = review_image_extraction(
        _image(tmp_path),
        event,
        _request(),
        _extraction(event_id="event_x", amount="2854.0", field="grand_total"),
        client=client,
    )
    assert client.calls == 0
    assert reviewed.accepted is True


def test_invalid_cash_field_does_not_call_verification(tmp_path: Path) -> None:
    event = make_event(
        event_id="event_x",
        category="transport",
        description="Taxi fare",
        amount=None,
        currency=Currency.USD,
    )
    client = FakeClient({})
    reviewed = review_image_extraction(
        _image(tmp_path),
        event,
        _request(),
        _extraction(
            event_id="event_x",
            amount="40.00",
            field="cash_tendered",
            currency=Currency.USD,
        ),
        client=client,
    )
    assert client.calls == 0
    assert reviewed.accepted is False
    assert reviewed.validation.status is ValidationStatus.INVALID
