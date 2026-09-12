"""Cache structured evidence facts. Never store secrets or sample labels."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from config import REPO_ROOT
from data.models import Currency
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
    ImageExtraction,
)

DEFAULT_CACHE_PATH = REPO_ROOT / "evaluation" / "cache" / "evidence.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return value


def cache_key(*, source_id: str, kind: str, version: str, model: str) -> str:
    return f"{kind}:{source_id}:{version}:{model}"


class EvidenceCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CACHE_PATH
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get_facts(self, key: str) -> tuple[EvidenceFact, ...] | None:
        raw = self._data.get(key)
        if raw is None:
            return None
        return tuple(_fact_from_dict(item) for item in raw.get("facts", []))

    def get_image(self, key: str) -> ImageExtraction | None:
        raw = self._data.get(key)
        if raw is None:
            return None
        return _image_from_dict(raw["image"])

    def put_facts(self, key: str, facts: tuple[EvidenceFact, ...]) -> None:
        self._data[key] = {
            "facts": [
                {field: _json_safe(value) for field, value in asdict(fact).items()}
                for fact in facts
            ]
        }
        self._save()

    def put_image(self, key: str, extraction: ImageExtraction) -> None:
        self._data[key] = {
            "image": {field: _json_safe(value) for field, value in asdict(extraction).items()}
        }
        self._save()

    def get_verification(self, key: str) -> tuple[ImageExtraction, bool | None, str] | None:
        raw = self._data.get(key)
        if raw is None or "image" not in raw:
            return None
        first_correct = raw.get("first_amount_correct")
        if isinstance(first_correct, str):
            first_correct = first_correct.strip().lower() in {"true", "1", "yes"}
        supporting = str(raw.get("supporting_text") or "")
        return _image_from_dict(raw["image"]), first_correct, supporting

    def put_verification(
        self,
        key: str,
        extraction: ImageExtraction,
        *,
        first_amount_correct: bool | None,
        supporting_text: str,
    ) -> None:
        self._data[key] = {
            "image": {field: _json_safe(value) for field, value in asdict(extraction).items()},
            "first_amount_correct": first_amount_correct,
            "supporting_text": supporting_text,
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")


def _fact_from_dict(item: dict[str, Any]) -> EvidenceFact:
    return EvidenceFact(
        source_type=EvidenceSourceType(item["source_type"]),
        source_id=item["source_id"],
        user_id=item["user_id"],
        request_id=item.get("request_id"),
        related_event_id=item.get("related_event_id"),
        fact_type=FactType(item["fact_type"]),
        amount=Decimal(item["amount"]) if item.get("amount") is not None else None,
        currency=Currency(item["currency"]) if item.get("currency") else None,
        effective_date=date.fromisoformat(item["effective_date"])
        if item.get("effective_date")
        else None,
        status=FactStatus(item["status"]),
        supersedes_event_id=item.get("supersedes_event_id"),
        confidence=EvidenceConfidence(item["confidence"]),
        extraction_method=ExtractionMethod.CACHE,
        raw_reference=item.get("raw_reference", ""),
        notes=item.get("notes", ""),
        percent=Decimal(item["percent"]) if item.get("percent") is not None else None,
        category=item.get("category"),
    )


def _image_from_dict(item: dict[str, Any]) -> ImageExtraction:
    return ImageExtraction(
        image_id=item["image_id"],
        related_event_id=item["related_event_id"],
        amount=Decimal(item["amount"]) if item.get("amount") is not None else None,
        currency=Currency(item["currency"]) if item.get("currency") else None,
        confidence=EvidenceConfidence(item["confidence"]),
        extraction_method=ExtractionMethod.CACHE,
        rationale=item.get("rationale", ""),
        selected_label=item.get("selected_label"),
        unresolved_reason=item.get("unresolved_reason"),
    )
