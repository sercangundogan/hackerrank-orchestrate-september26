from __future__ import annotations

from decimal import Decimal

from ai.client import ModelClient
from decision.capacity import compute_capacity
from evidence.cache import EvidenceCache
from evidence.pipeline import resolve_request_state
from finance.forecast_models import ForecastConfig
from finance.normalization import finance_request_by_id


def test_resolved_real_image_cases_have_no_unresolved_capacity(repository) -> None:
    cache = EvidenceCache()
    client = ModelClient(api_key="unused")
    config = ForecastConfig(strict_unresolved_amounts=True)
    for request_id in (
        "request_16",
        "request_20",
        "request_64",
        "request_73",
    ):
        request = finance_request_by_id(repository, request_id)
        _, state, _ = resolve_request_state(
            repository, request, client=client, cache=cache
        )
        result = compute_capacity(state, request, config)
        assert result.unresolved_reasons == ()
        assert result.amount_safe_to_pay >= Decimal("0")
        assert result.amount_safe_to_pay <= request.requested_amount
