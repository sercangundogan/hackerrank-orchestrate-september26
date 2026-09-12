"""Central model-price table. Amounts are USD per token.

Override with environment variables if the contest host publishes different
list prices. Do not scatter cost arithmetic elsewhere.
"""

from __future__ import annotations

import os
from decimal import Decimal


# Published OpenAI list prices used as the default estimate (USD / token).
_DEFAULT = {
    "gpt-4o-mini": (Decimal("0.00000015"), Decimal("0.00000060")),
    "gpt-4o": (Decimal("0.00000250"), Decimal("0.00001000")),
    "gpt-4.1-mini": (Decimal("0.00000040"), Decimal("0.00000160")),
}


def _env_price(name: str, fallback: Decimal) -> Decimal:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    return Decimal(raw)


def prices_for_model(model: str) -> tuple[Decimal, Decimal]:
    key = model.strip()
    input_price, output_price = _DEFAULT.get(key, (Decimal("0.00000015"), Decimal("0.00000060")))
    input_price = _env_price("BUYORWAIT_PRICE_INPUT_PER_TOKEN", input_price)
    output_price = _env_price("BUYORWAIT_PRICE_OUTPUT_PER_TOKEN", output_price)
    return input_price, output_price


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    input_price, output_price = prices_for_model(model)
    return (input_price * input_tokens + output_price * output_tokens).quantize(Decimal("0.000001"))
