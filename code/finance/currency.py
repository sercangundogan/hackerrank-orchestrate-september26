"""Exact dated FX conversion. No inverses, no neighboring dates, no live rates."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol

from data.models import Currency, ExchangeRate


class RateLookup(Protocol):
    def exchange_rate(
        self,
        rate_date: date,
        from_currency: Currency,
        to_currency: Currency,
    ) -> ExchangeRate: ...


class MissingExchangeRateError(LookupError):
    """No exact dated rate exists for the requested pair."""


def convert_to_home(
    amount: Decimal,
    from_currency: Currency,
    home_currency: Currency,
    rate_date: date,
    rates: RateLookup,
) -> Decimal:
    """Convert `amount` into `home_currency` using an exact dated direct rate.

    Same-currency amounts are returned unchanged. Inverse pairs and other
    dates are never used. Full Decimal precision is preserved; callers must
    not treat this as a rounded output amount.
    """
    if amount is None:
        raise ValueError("cannot convert a missing amount")
    if from_currency is home_currency:
        return amount
    try:
        quoted = rates.exchange_rate(rate_date, from_currency, home_currency)
    except LookupError as exc:
        raise MissingExchangeRateError(
            "missing exact exchange rate for "
            f"date={rate_date.isoformat()} "
            f"from={from_currency.value} to={home_currency.value}"
        ) from exc
    return amount * quoted.rate
