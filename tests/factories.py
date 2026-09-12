from __future__ import annotations

from datetime import date
from decimal import Decimal

from data.models import (
    Currency,
    EventDirection,
    EventStatus,
    EventType,
    ExchangeRate,
    FinancialEvent,
    Flexibility,
)


class FakeRateLookup:
    def __init__(self, rates: dict[tuple[date, Currency, Currency], Decimal] | None = None) -> None:
        self.rates = rates or {}

    def exchange_rate(
        self,
        rate_date: date,
        from_currency: Currency,
        to_currency: Currency,
    ) -> ExchangeRate:
        key = (rate_date, from_currency, to_currency)
        if key not in self.rates:
            raise LookupError(
                f"no exact exchange rate for {rate_date} {from_currency.value}->{to_currency.value}"
            )
        return ExchangeRate(rate_date, from_currency, to_currency, self.rates[key])


def make_event(**overrides: object) -> FinancialEvent:
    values: dict[str, object] = {
        "event_id": "event_x",
        "user_id": "user_01",
        "event_type": EventType.EXPENSE,
        "description": "Test expense",
        "category": "rent",
        "direction": EventDirection.DEBIT,
        "amount": Decimal("10"),
        "currency": Currency.ZAR,
        "event_date": date(2024, 1, 1),
        "settlement_date": date(2024, 1, 1),
        "status": EventStatus.SETTLED,
        "linked_event_id": None,
        "flexibility": Flexibility.FIXED,
        "minimum_allowed_amount": None,
    }
    values.update(overrides)
    return FinancialEvent(**values)  # type: ignore[arg-type]
