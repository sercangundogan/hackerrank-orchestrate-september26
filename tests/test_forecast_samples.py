"""Sample-request forecast diagnostics.

These tests inspect simulator behavior. They do not read or assert sample
decision labels.
"""

from __future__ import annotations

from datetime import timedelta

from finance.forecast import forecast_financial_state
from finance.forecast_models import ForecastConfig, ForecastEventKind
from finance.normalization import finance_request_by_id, normalize_financial_state


def _forecast(repository, request_id: str):
    request = finance_request_by_id(repository, request_id)
    state = normalize_financial_state(repository, request)
    result = forecast_financial_state(state, ForecastConfig())
    return state, result


def test_sample_forecasts_do_not_replay_historical_cash(repository) -> None:
    for request_id in (
        "request_01",
        "request_02",
        "request_03",
        "request_06",
        "request_11",
        "request_12",
        "request_19",
        "request_21",
        "request_25",
    ):
        state, result = _forecast(repository, request_id)
        assert result.opening_balance == state.opening_balance
        assert result.opening_balance == state.profile.current_available_balance
        assert result.horizon_end == state.request_date + timedelta(days=90)
        historical_ids = {event.source_event_id for event in state.historical_settled_cash_events}
        applied_ids = {
            entry.source_event_id
            for entry in result.all_entries
            if entry.source_event_id and not entry.is_generated_recurrence
        }
        assert historical_ids.isdisjoint(applied_ids)


def test_request_01_reserves_pending_debit_and_uses_scheduled_salary(repository) -> None:
    state, result = _forecast(repository, "request_01")
    assert state.pending_debits
    assert any(
        entry.kind.value == "pending_debit_reserve" for entry in result.all_entries
    )
    assert all(entry.date == state.request_date for entry in result.all_entries if entry.kind.value == "pending_debit_reserve")
    assert state.confirmed_scheduled_income
    salary_ids = {event.source_event_id for event in state.confirmed_scheduled_income}
    applied = {entry.source_event_id for entry in result.all_entries}
    assert salary_ids <= applied


def test_request_02_projects_regular_salary_and_flags_message(repository) -> None:
    state, result = _forecast(repository, "request_02")
    generated_salary = [
        entry
        for entry in result.all_entries
        if entry.is_generated_recurrence and entry.category == "salary"
    ]
    assert generated_salary
    assert result.message_uncertainty_flags
    assert not state.confirmed_scheduled_income


def test_request_03_blank_salary_is_unresolved_not_zero(repository) -> None:
    state, result = _forecast(repository, "request_03")
    assert state.unresolved_events
    assert result.unresolved_reasons
    assert all(
        entry.amount_home_currency is not None and entry.amount_home_currency != 0
        or entry.amount_home_currency != 0
        for entry in result.all_entries
    )
    blank_ids = {
        event.source_event_id
        for event in state.unresolved_events
        if event.requires_external_evidence
    }
    applied = {entry.source_event_id for entry in result.all_entries}
    assert blank_ids.isdisjoint(applied)


def test_request_11_does_not_auto_forecast_low_irregular_dining(repository) -> None:
    state, result = _forecast(repository, "request_11")
    dining_series = [
        series
        for series in state.recurring_series_candidates
        if series.category == "dining"
    ]
    assert dining_series
    generated_dining = [
        entry
        for entry in result.all_entries
        if entry.category == "dining" and entry.is_generated_recurrence
    ]
    assert generated_dining == []


def test_request_12_does_not_invent_salary_after_low_history(repository) -> None:
    state, result = _forecast(repository, "request_12")
    generated_salary = [
        entry
        for entry in result.all_entries
        if entry.category == "salary" and entry.is_generated_recurrence
    ]
    assert generated_salary == []
    assert state.requires_message_resolution
    assert result.message_uncertainty_flags


def test_request_21_keeps_flexible_recurring_in_baseline(repository) -> None:
    _, result = _forecast(repository, "request_21")
    generated = [entry for entry in result.all_entries if entry.is_generated_recurrence]
    categories = {entry.category for entry in generated}
    assert "salary" in categories or any(
        entry.kind is ForecastEventKind.CONFIRMED_CREDIT for entry in result.all_entries
    )
    # Flexible subscriptions remain until a later spending-change phase.
    assert any(
        entry.category in {"streaming", "cloud_storage", "software"}
        or (entry.flexibility is not None and entry.flexibility.value != "fixed")
        for entry in generated
    )


def test_request_25_fx_salary_uses_home_currency_amounts(repository) -> None:
    state, result = _forecast(repository, "request_25")
    salary = [
        entry
        for entry in result.all_entries
        if entry.category == "salary"
    ]
    assert salary
    assert all(entry.amount_home_currency > 0 for entry in salary)
    assert state.profile.home_currency.value == "IDR"
