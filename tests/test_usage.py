from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from evidence.cache import EvidenceCache, cache_key
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
)
from usage.pricing import estimate_cost
from usage.tracker import UsageTracker


def _fact() -> EvidenceFact:
    return EvidenceFact(
        source_type=EvidenceSourceType.MESSAGE,
        source_id="message_01",
        user_id="user_02",
        request_id="request_02",
        related_event_id=None,
        fact_type=FactType.SALARY_AMOUNT_CHANGE,
        amount=Decimal("1"),
        currency=None,
        effective_date=None,
        status=FactStatus.ACTIVE,
        supersedes_event_id=None,
        confidence=EvidenceConfidence.HIGH,
        extraction_method=ExtractionMethod.DETERMINISTIC,
        raw_reference="message_01",
        notes="cached",
    )


def test_cost_uses_central_pricing() -> None:
    cost = estimate_cost("gpt-4o-mini", 1000, 500)
    assert cost > Decimal("0")
    assert cost == Decimal("0.000150") + Decimal("0.000300")


def test_usage_records_tokens_and_failed_calls() -> None:
    tracker = UsageTracker()
    tracker.record(
        provider="openai",
        model="gpt-4o-mini",
        purpose="message_extract",
        request_id="request_02",
        user_id="user_02",
        source_id="message_01",
        prompt_version="v1",
        input_tokens=20,
        output_tokens=10,
        cache_hit=False,
        success=True,
    )
    tracker.record(
        provider="openai",
        model="gpt-4o-mini",
        purpose="image_amount_extract",
        request_id="request_03",
        user_id="user_03",
        source_id="image_01",
        prompt_version="v1",
        input_tokens=0,
        output_tokens=0,
        cache_hit=False,
        success=False,
        error="timeout",
    )
    summary = tracker.summary()
    assert summary["calls"] == 2
    assert summary["paid_calls"] == 1
    assert summary["failures"] == 1
    assert summary["input_tokens"] == 20
    assert summary["output_tokens"] == 10


def test_cache_hit_does_not_count_as_paid_call(tmp_path: Path) -> None:
    cache = EvidenceCache(tmp_path / "cache.json")
    key = cache_key(source_id="message_01", kind="message", version="v1", model="none")
    cache.put_facts(key, (_fact(),))
    tracker = UsageTracker()
    cached = cache.get_facts(key)
    assert cached is not None
    tracker.record(
        provider="cache",
        model="none",
        purpose="message_extract",
        request_id="request_02",
        user_id="user_02",
        source_id="message_01",
        prompt_version="v1",
        input_tokens=0,
        output_tokens=0,
        cache_hit=True,
        success=True,
    )
    summary = tracker.summary()
    assert summary["cache_hits"] == 1
    assert summary["paid_calls"] == 0
    assert Decimal(str(summary["estimated_cost_usd"])) == Decimal("0")
