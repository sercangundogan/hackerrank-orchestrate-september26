"""Record every model call. Never store secrets or sample decision labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from config import REPO_ROOT
from usage.pricing import estimate_cost

USAGE_JSONL = REPO_ROOT / "evaluation" / "usage.jsonl"


@dataclass
class UsageRecord:
    timestamp: str
    provider: str
    model: str
    purpose: str
    request_id: str | None
    user_id: str | None
    source_id: str | None
    prompt_version: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: str
    cache_hit: bool
    success: bool
    error: str | None = None


@dataclass
class UsageTracker:
    records: list[UsageRecord] = field(default_factory=list)
    persist_path: Path = USAGE_JSONL

    def record(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        request_id: str | None,
        user_id: str | None,
        source_id: str | None,
        prompt_version: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool,
        success: bool,
        error: str | None = None,
    ) -> UsageRecord:
        cost = estimate_cost(model, input_tokens, output_tokens) if success and not cache_hit else Decimal("0")
        item = UsageRecord(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            provider=provider,
            model=model,
            purpose=purpose,
            request_id=request_id,
            user_id=user_id,
            source_id=source_id,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=str(cost),
            cache_hit=cache_hit,
            success=success,
            error=error,
        )
        self.records.append(item)
        return item

    def persist(self) -> None:
        if not self.records:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("a", encoding="utf-8") as handle:
            for item in self.records:
                handle.write(json.dumps(asdict(item), ensure_ascii=True) + "\n")

    def persist_overwrite(self) -> None:
        """Replace the ledger so a final-run file contains only this run."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("w", encoding="utf-8") as handle:
            for item in self.records:
                handle.write(json.dumps(asdict(item), ensure_ascii=True) + "\n")

    def summary(self) -> dict[str, object]:
        paid = [item for item in self.records if item.success and not item.cache_hit]
        total_in = sum(item.input_tokens for item in paid)
        total_out = sum(item.output_tokens for item in paid)
        total_cost = sum((Decimal(item.estimated_cost_usd) for item in paid), Decimal("0"))
        return {
            "calls": len(self.records),
            "paid_calls": len(paid),
            "cache_hits": sum(1 for item in self.records if item.cache_hit),
            "failures": sum(1 for item in self.records if not item.success),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": str(total_cost),
        }
