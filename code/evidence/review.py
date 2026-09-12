"""Review image extractions without mutating the raw first-pass VLM response."""

from __future__ import annotations

import base64
from decimal import Decimal, InvalidOperation

from ai.client import ModelClient, ModelError, ModelUnavailableError
from ai.schemas import (
    IMAGE_VERIFICATION_PROMPT_VERSION,
    IMAGE_VERIFICATION_SCHEMA,
    IMAGE_VERIFICATION_SYSTEM_PROMPT,
)
from data.models import FinancialEvent, FinanceRequest, ImageReference
from evidence.cache import EvidenceCache, cache_key
from evidence.image_parser import _validate_image_payload, image_to_fact
from evidence.models import (
    EvidenceFact,
    FactStatus,
    ImageExtraction,
    ImageVerification,
    ReviewedImageExtraction,
    ValidationStatus,
)
from evidence.semantic import amounts_equivalent, normalize_field, validate_image_semantics
from usage.tracker import UsageTracker


def _boolish(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _candidate(label: str, amount: Decimal | None) -> str:
    amount_text = str(amount) if amount is not None else "none"
    return f"{label}:{amount_text}"


def _verification_user_prompt(
    event: FinancialEvent,
    extraction: ImageExtraction,
    reason: str,
) -> str:
    first_amount = str(extraction.amount) if extraction.amount is not None else "null"
    first_field = extraction.selected_label or "null"
    first_currency = extraction.currency.value if extraction.currency else "null"
    return (
        f"event_id={event.event_id}\n"
        f"event_type={event.event_type.value}\n"
        f"description={event.description}\n"
        f"category={event.category}\n"
        f"direction={event.direction.value}\n"
        f"currency={event.currency.value}\n"
        f"first_extraction_amount={first_amount}\n"
        f"first_extraction_currency={first_currency}\n"
        f"first_extraction_semantic_field={first_field}\n"
        f"first_extraction_confidence={extraction.confidence.value}\n"
        f"semantic_inconsistency_reason={reason}\n"
        "The first extraction failed a deterministic semantic-consistency check. "
        "Identify the document field that represents this event and its exact amount. "
        "Say whether the first extraction amount is still correct even if its semantic label was wrong. "
        "Do not evaluate affordability."
    )


def verify_image_extraction(
    image: ImageReference,
    event: FinancialEvent,
    request: FinanceRequest,
    extraction: ImageExtraction,
    reason: str,
    *,
    client: ModelClient | None = None,
    cache: EvidenceCache | None = None,
    tracker: UsageTracker | None = None,
) -> ImageVerification | None:
    model_name = client.vision_model if client is not None else "none"
    first_field = normalize_field(extraction.selected_label) or "none"
    first_amount = str(extraction.amount) if extraction.amount is not None else "none"
    key = cache_key(
        source_id=f"{image.image_id}:{first_field}:{first_amount}",
        kind="image_verify",
        version=IMAGE_VERIFICATION_PROMPT_VERSION,
        model=model_name,
    )
    if cache is not None:
        cached = cache.get_verification(key)
        if cached is not None:
            cached_extraction, first_correct, supporting = cached
            if tracker is not None:
                tracker.record(
                    provider="cache",
                    model=model_name,
                    purpose="image_amount_verify",
                    request_id=request.request_id,
                    user_id=image.user_id,
                    source_id=image.image_id,
                    prompt_version=IMAGE_VERIFICATION_PROMPT_VERSION,
                    input_tokens=0,
                    output_tokens=0,
                    cache_hit=True,
                    success=True,
                )
            return ImageVerification(
                extraction=cached_extraction,
                first_amount_correct=first_correct,
                supporting_text=supporting,
            )

    if client is None or not client.available():
        return None

    image_bytes = image.path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        response = client.complete_json(
            system_prompt=IMAGE_VERIFICATION_SYSTEM_PROMPT,
            user_text=_verification_user_prompt(event, extraction, reason),
            purpose="image_amount_verify",
            prompt_version=IMAGE_VERIFICATION_PROMPT_VERSION,
            request_id=request.request_id,
            user_id=image.user_id,
            source_id=image.image_id,
            image_media_type="image/png",
            image_b64=image_b64,
            schema=IMAGE_VERIFICATION_SCHEMA,
        )
        verified = _validate_image_payload(response.payload, event)
        verified = ImageExtraction(
            image_id=image.image_id,
            related_event_id=verified.related_event_id,
            amount=verified.amount,
            currency=verified.currency,
            confidence=verified.confidence,
            extraction_method=verified.extraction_method,
            rationale=verified.rationale,
            selected_label=verified.selected_label,
            unresolved_reason=verified.unresolved_reason,
        )
        first_correct = _boolish(response.payload.get("first_amount_correct"))
        supporting = str(response.payload.get("supporting_text") or "")
    except (ModelUnavailableError, ModelError, ValueError, InvalidOperation, KeyError):
        return None

    result = ImageVerification(
        extraction=verified,
        first_amount_correct=first_correct,
        supporting_text=supporting,
    )
    if cache is not None and verified.amount is not None:
        cache.put_verification(
            key,
            verified,
            first_amount_correct=first_correct,
            supporting_text=supporting,
        )
    return result


def review_image_extraction(
    image: ImageReference,
    event: FinancialEvent,
    request: FinanceRequest,
    extraction: ImageExtraction,
    *,
    client: ModelClient | None = None,
    cache: EvidenceCache | None = None,
    tracker: UsageTracker | None = None,
) -> ReviewedImageExtraction:
    validation = validate_image_semantics(event, extraction)
    original_field = normalize_field(extraction.selected_label)
    if validation.status is ValidationStatus.VALID:
        return ReviewedImageExtraction(
            original=extraction,
            validation=validation,
            final_amount=extraction.amount,
            final_field=original_field,
            accepted=True,
        )

    verification = None
    if validation.status is ValidationStatus.NEEDS_REVIEW:
        verification = verify_image_extraction(
            image,
            event,
            request,
            extraction,
            validation.validation_reason,
            client=client,
            cache=cache,
            tracker=tracker,
        )

    if verification is None:
        reason = (
            "targeted verification unavailable"
            if validation.status is ValidationStatus.NEEDS_REVIEW
            else validation.validation_reason
        )
        return ReviewedImageExtraction(
            original=extraction,
            validation=validation,
            accepted=False,
            unresolved_reason=reason,
            candidates=(_candidate(original_field or "unknown", extraction.amount),),
        )

    verified = verification.extraction
    verified_validation = validate_image_semantics(event, verified)
    verified_field = normalize_field(verified.selected_label)
    first_amount = extraction.amount
    second_amount = verified.amount
    candidates = (
        _candidate(original_field or "unknown", first_amount),
        _candidate(verified_field or "unknown", second_amount),
    )

    if verified.amount is None or verified_validation.status is not ValidationStatus.VALID:
        return ReviewedImageExtraction(
            original=extraction,
            validation=validation,
            verification=verification,
            accepted=False,
            unresolved_reason=(
                verified_validation.validation_reason
                if verified.amount is not None
                else "verification did not resolve a compatible amount"
            ),
            candidates=candidates,
        )

    same_amount = amounts_equivalent(first_amount, second_amount)
    if same_amount:
        return ReviewedImageExtraction(
            original=extraction,
            validation=validation,
            verification=verification,
            final_amount=first_amount,
            final_field=verified_field,
            accepted=True,
            candidates=candidates,
        )
    if verification.first_amount_correct is True and first_amount is not None:
        return ReviewedImageExtraction(
            original=extraction,
            validation=validation,
            verification=verification,
            final_amount=first_amount,
            final_field=verified_field,
            accepted=True,
            candidates=candidates,
        )
    return ReviewedImageExtraction(
        original=extraction,
        validation=validation,
        verification=verification,
        accepted=False,
        unresolved_reason=(
            "first extraction and verification amounts disagree; "
            "both candidates retained"
        ),
        candidates=candidates,
    )


def reviewed_image_to_fact(
    image: ImageReference,
    reviewed: ReviewedImageExtraction,
) -> EvidenceFact:
    fact = image_to_fact(image, reviewed.original)
    notes = reviewed.original.rationale
    if reviewed.verification is not None:
        notes = (
            f"{notes} | verification={reviewed.verification.extraction.rationale} "
            f"field={reviewed.final_field or reviewed.verification.extraction.selected_label}"
        )
    notes = f"{notes} | validation={reviewed.validation.status.value}: {reviewed.validation.validation_reason}"
    if reviewed.accepted:
        return EvidenceFact(
            source_type=fact.source_type,
            source_id=fact.source_id,
            user_id=fact.user_id,
            request_id=fact.request_id,
            related_event_id=fact.related_event_id,
            fact_type=fact.fact_type,
            amount=reviewed.final_amount,
            currency=fact.currency or reviewed.original.currency,
            effective_date=fact.effective_date,
            status=FactStatus.ACTIVE,
            supersedes_event_id=fact.supersedes_event_id,
            confidence=fact.confidence,
            extraction_method=fact.extraction_method,
            raw_reference=fact.raw_reference,
            notes=notes,
        )
    return EvidenceFact(
        source_type=fact.source_type,
        source_id=fact.source_id,
        user_id=fact.user_id,
        request_id=fact.request_id,
        related_event_id=fact.related_event_id,
        fact_type=fact.fact_type,
        amount=None,
        currency=fact.currency,
        effective_date=fact.effective_date,
        status=FactStatus.PENDING,
        supersedes_event_id=fact.supersedes_event_id,
        confidence=fact.confidence,
        extraction_method=fact.extraction_method,
        raw_reference=fact.raw_reference,
        notes=f"{notes} | unresolved={reviewed.unresolved_reason}",
    )
