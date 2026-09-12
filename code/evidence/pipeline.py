"""Collect message and image facts for one request without deciding affordability."""

from __future__ import annotations

from data.models import FinanceRequest
from data.repository import DatasetRepository
from evidence.cache import EvidenceCache
from evidence.image_parser import parse_image
from evidence.message_parser import needs_llm, parse_message, extract_deterministic
from evidence.models import EvidenceBundle, EvidenceFact, ImageExtraction, ReviewedImageExtraction
from evidence.review import review_image_extraction, reviewed_image_to_fact
from evidence.resolver import resolve_financial_state
from finance.models import NormalizedFinancialState
from finance.normalization import normalize_financial_state
from ai.client import ModelClient
from usage.tracker import UsageTracker


def extract_evidence(
    repository: DatasetRepository,
    request: FinanceRequest,
    *,
    client: ModelClient | None = None,
    cache: EvidenceCache | None = None,
    tracker: UsageTracker | None = None,
) -> EvidenceBundle:
    messages = repository.messages_for_user(request.user_id)
    images = repository.images_for_user(request.user_id)
    events_by_id = {event.event_id: event for event in repository.events_for_user(request.user_id)}
    facts: list[EvidenceFact] = []
    llm_ids: list[str] = []
    vlm_ids: list[str] = []
    failures: list[str] = []
    image_rows: list[ImageExtraction] = []
    image_reviews: list[ReviewedImageExtraction] = []

    for message in messages:
        related = events_by_id.get(message.related_event_id) if message.related_event_id else None
        extracted = parse_message(
            message,
            related_event=related,
            request=request,
            client=client,
            cache=cache,
            tracker=tracker,
        )
        facts.extend(extracted)
        if needs_llm(message, list(extract_deterministic(message))):
            llm_ids.append(message.message_id)
        if not extracted and needs_llm(message, []):
            failures.append(f"{message.message_id}: no structured facts extracted")

    for image in images:
        event = events_by_id.get(image.related_event_id)
        if event is None:
            failures.append(f"{image.image_id}: missing related event {image.related_event_id}")
            continue
        extraction = parse_image(
            image,
            event,
            request,
            client=client,
            cache=cache,
            tracker=tracker,
        )
        reviewed = review_image_extraction(
            image,
            event,
            request,
            extraction,
            client=client,
            cache=cache,
            tracker=tracker,
        )
        image_rows.append(extraction)
        image_reviews.append(reviewed)
        vlm_ids.append(image.image_id)
        facts.append(reviewed_image_to_fact(image, reviewed))
        if not reviewed.accepted:
            failures.append(
                f"{image.image_id}: unresolved "
                f"({reviewed.unresolved_reason or extraction.unresolved_reason or 'no amount'})"
            )

    return EvidenceBundle(
        request_id=request.request_id,
        user_id=request.user_id,
        facts=tuple(facts),
        image_extractions=tuple(image_rows),
        llm_source_ids=tuple(llm_ids),
        vlm_source_ids=tuple(vlm_ids),
        failures=tuple(failures),
        image_reviews=tuple(image_reviews),
    )


def resolve_request_state(
    repository: DatasetRepository,
    request: FinanceRequest,
    *,
    client: ModelClient | None = None,
    cache: EvidenceCache | None = None,
    tracker: UsageTracker | None = None,
) -> tuple[NormalizedFinancialState, NormalizedFinancialState, EvidenceBundle]:
    base = normalize_financial_state(repository, request)
    bundle = extract_evidence(
        repository,
        request,
        client=client,
        cache=cache,
        tracker=tracker,
    )
    resolved = resolve_financial_state(
        base,
        bundle.facts,
        repository=repository,
        messages=repository.messages_for_user(request.user_id),
    )
    return base, resolved, bundle
