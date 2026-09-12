"""Image → structured event amount. Never treat failure as zero."""

from __future__ import annotations

import base64
from decimal import Decimal, InvalidOperation

from ai.client import ModelClient, ModelError, ModelUnavailableError
from ai.schemas import (
    IMAGE_AMOUNT_PROMPT_VERSION,
    IMAGE_RESPONSE_SCHEMA,
    IMAGE_SYSTEM_PROMPT,
)
from data.models import Currency, FinancialEvent, FinanceRequest, ImageReference
from evidence.cache import EvidenceCache, cache_key
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
    ImageExtraction,
)
from usage.tracker import UsageTracker


def _user_prompt(image: ImageReference, event: FinancialEvent, request: FinanceRequest) -> str:
    return (
        f"image_id={image.image_id}\n"
        f"related_event_id={event.event_id}\n"
        f"event_type={event.event_type.value}\n"
        f"category={event.category}\n"
        f"description={event.description}\n"
        f"direction={event.direction.value}\n"
        f"status={event.status.value}\n"
        f"currency={event.currency.value}\n"
        f"request_id={request.request_id}\n"
        f"request_date={request.request_date.isoformat()}\n"
        "Extract only the amount that belongs to this event."
    )


def _validate_image_payload(payload: dict, event: FinancialEvent) -> ImageExtraction:
    related = payload.get("related_event_id")
    if related != event.event_id:
        raise ValueError("model invented or mismatched related_event_id")
    amount_raw = payload.get("amount")
    amount = None
    if amount_raw not in (None, ""):
        amount = Decimal(str(amount_raw))
        if amount < 0:
            raise ValueError("amount must be non-negative")
    currency_raw = payload.get("currency")
    currency = Currency(currency_raw) if currency_raw else None
    confidence = EvidenceConfidence(payload.get("confidence", "low"))
    rationale = str(payload.get("rationale") or "")
    if amount is None:
        return ImageExtraction(
            image_id="",
            related_event_id=event.event_id,
            amount=None,
            currency=currency,
            confidence=confidence,
            extraction_method=ExtractionMethod.VLM,
            rationale=rationale,
            selected_label=payload.get("selected_label"),
            unresolved_reason=rationale or "amount unreadable",
        )
    return ImageExtraction(
        image_id="",
        related_event_id=event.event_id,
        amount=amount,
        currency=currency or event.currency,
        confidence=confidence,
        extraction_method=ExtractionMethod.VLM,
        rationale=rationale,
        selected_label=payload.get("selected_label"),
    )


def parse_image(
    image: ImageReference,
    event: FinancialEvent,
    request: FinanceRequest,
    *,
    client: ModelClient | None = None,
    cache: EvidenceCache | None = None,
    tracker: UsageTracker | None = None,
) -> ImageExtraction:
    model_name = client.vision_model if client is not None else "none"
    key = cache_key(
        source_id=image.image_id,
        kind="image",
        version=IMAGE_AMOUNT_PROMPT_VERSION,
        model=model_name,
    )
    if cache is not None:
        cached = cache.get_image(key)
        if cached is not None:
            if tracker is not None:
                tracker.record(
                    provider="cache",
                    model=model_name,
                    purpose="image_amount_extract",
                    request_id=request.request_id,
                    user_id=image.user_id,
                    source_id=image.image_id,
                    prompt_version=IMAGE_AMOUNT_PROMPT_VERSION,
                    input_tokens=0,
                    output_tokens=0,
                    cache_hit=True,
                    success=True,
                )
            return cached

    if client is None or not client.available():
        return ImageExtraction(
            image_id=image.image_id,
            related_event_id=event.event_id,
            amount=None,
            currency=None,
            confidence=EvidenceConfidence.LOW,
            extraction_method=ExtractionMethod.VLM,
            rationale="vision model unavailable",
            unresolved_reason="vision model unavailable",
        )

    image_bytes = image.path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        response = client.complete_json(
            system_prompt=IMAGE_SYSTEM_PROMPT,
            user_text=_user_prompt(image, event, request),
            purpose="image_amount_extract",
            prompt_version=IMAGE_AMOUNT_PROMPT_VERSION,
            request_id=request.request_id,
            user_id=image.user_id,
            source_id=image.image_id,
            image_media_type="image/png",
            image_b64=image_b64,
            schema=IMAGE_RESPONSE_SCHEMA,
        )
        extraction = _validate_image_payload(response.payload, event)
    except (ModelUnavailableError, ModelError, ValueError, InvalidOperation, KeyError) as exc:
        return ImageExtraction(
            image_id=image.image_id,
            related_event_id=event.event_id,
            amount=None,
            currency=None,
            confidence=EvidenceConfidence.LOW,
            extraction_method=ExtractionMethod.VLM,
            rationale="extraction failed",
            unresolved_reason=str(exc),
        )
    extraction = ImageExtraction(
        image_id=image.image_id,
        related_event_id=extraction.related_event_id,
        amount=extraction.amount,
        currency=extraction.currency,
        confidence=extraction.confidence,
        extraction_method=extraction.extraction_method,
        rationale=extraction.rationale,
        selected_label=extraction.selected_label,
        unresolved_reason=extraction.unresolved_reason,
    )
    if cache is not None and extraction.amount is not None:
        cache.put_image(key, extraction)
    return extraction


def image_to_fact(image: ImageReference, extraction: ImageExtraction) -> EvidenceFact:
    return EvidenceFact(
        source_type=EvidenceSourceType.IMAGE,
        source_id=image.image_id,
        user_id=image.user_id,
        request_id=image.request_id,
        related_event_id=extraction.related_event_id,
        fact_type=FactType.EVENT_AMOUNT,
        amount=extraction.amount,
        currency=extraction.currency,
        effective_date=None,
        status=FactStatus.ACTIVE if extraction.amount is not None else FactStatus.PENDING,
        supersedes_event_id=extraction.related_event_id,
        confidence=extraction.confidence,
        extraction_method=extraction.extraction_method,
        raw_reference=image.image_id,
        notes=extraction.rationale,
    )
