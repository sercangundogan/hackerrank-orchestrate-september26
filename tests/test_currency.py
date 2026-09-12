from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from data.models import Currency
from finance.currency import MissingExchangeRateError, convert_to_home
from tests.factories import FakeRateLookup


def test_same_currency_unchanged() -> None:
    rates = FakeRateLookup()
    amount = Decimal("123.45")
    assert (
        convert_to_home(amount, Currency.ZAR, Currency.ZAR, date(2024, 1, 1), rates)
        is amount
        or convert_to_home(amount, Currency.ZAR, Currency.ZAR, date(2024, 1, 1), rates)
        == amount
    )
    converted = convert_to_home(amount, Currency.EUR, Currency.EUR, date(2023, 10, 15), rates)
    assert converted == Decimal("123.45")
    assert isinstance(converted, Decimal)


def test_direct_fx_conversion_preserves_decimal() -> None:
    rates = FakeRateLookup(
        {(date(2023, 10, 15), Currency.USD, Currency.EUR): Decimal("0.92")}
    )
    converted = convert_to_home(
        Decimal("1800"), Currency.USD, Currency.EUR, date(2023, 10, 15), rates
    )
    assert converted == Decimal("1656.00")
    assert converted == Decimal("1800") * Decimal("0.92")


def test_does_not_invert_or_use_another_date() -> None:
    rates = FakeRateLookup(
        {
            (date(2023, 10, 15), Currency.USD, Currency.EUR): Decimal("0.92"),
            (date(2023, 11, 15), Currency.EUR, Currency.USD): Decimal("1.09"),
        }
    )
    with pytest.raises(MissingExchangeRateError, match="missing exact exchange rate"):
        convert_to_home(Decimal("10"), Currency.EUR, Currency.USD, date(2023, 10, 15), rates)
    with pytest.raises(MissingExchangeRateError):
        convert_to_home(Decimal("10"), Currency.USD, Currency.EUR, date(2023, 10, 14), rates)
