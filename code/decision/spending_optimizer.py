"""Legal optional spending-change search after baseline capacity.

amount_safe_to_pay is never recomputed. Changes only affect the
counterfactual forecast used to validate a payment schedule.

Eligible events are active recurring expense series that are flexible,
unprotected, and listed in the user's reduce/stop preferences. The event id
written to output is the latest observation in that series — not a
hardcoded sample id.

Search is exhaustive over legal actions up to depth 3 after pruning series
that do not appear as future forecast debits. Combinations that both stop
and reduce the same series are discarded.
"""

from __future__ import annotations

from itertools import combinations

from data.models import EventDirection, Flexibility
from decision.models import SpendingAction, SpendingActionKind
from finance.forecast import series_key
from finance.forecast_models import ForecastResult
from finance.models import RecurringSeriesCandidate, NormalizedFinancialState

_FLEXIBLE = {
    Flexibility.REDUCIBLE,
    Flexibility.STOPPABLE,
    Flexibility.REDUCIBLE_OR_STOPPABLE,
}
_STOPPABLE = {Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE}
_REDUCIBLE = {Flexibility.REDUCIBLE, Flexibility.REDUCIBLE_OR_STOPPABLE}
MAX_SPENDING_CHANGES = 3


def latest_series_event_id(series: RecurringSeriesCandidate) -> str:
    """Generic latest-observation id for an active recurring series."""
    if series.observed_dates and len(series.event_ids) == len(series.observed_dates):
        index = max(range(len(series.observed_dates)), key=lambda i: series.observed_dates[i])
        return series.event_ids[index]
    return series.event_ids[-1]


def _normal_amount(series: RecurringSeriesCandidate):
    if series.representative_amount is not None:
        return series.representative_amount
    known = [amount for amount in series.observed_amounts_home_currency if amount is not None]
    if not known:
        return None
    return known[-1]


def series_has_future_debit(
    series: RecurringSeriesCandidate,
    forecast: ForecastResult,
) -> bool:
    key = series_key(series)
    return any(
        entry.source_series_key == key and entry.signed_amount < 0
        for entry in forecast.all_entries
    )


def legal_actions(
    state: NormalizedFinancialState,
    forecast: ForecastResult,
) -> tuple[SpendingAction, ...]:
    protected = set(state.profile.expense_categories_to_protect)
    reduce_ok = set(state.profile.expense_categories_user_is_willing_to_reduce)
    stop_ok = set(state.profile.expense_categories_user_is_willing_to_stop)
    actions: list[SpendingAction] = []
    for series in state.recurring_series_candidates:
        if series.direction is not EventDirection.DEBIT:
            continue
        if not series.event_ids:
            continue
        if series.flexibility not in _FLEXIBLE:
            continue
        if series.is_protected_category or series.category in protected:
            continue
        if not series_has_future_debit(series, forecast):
            continue
        normal = _normal_amount(series)
        if normal is None:
            continue
        event_id = latest_series_event_id(series)
        key = series_key(series)
        allows_stop = series.profile_allows_stop or series.category in stop_ok
        allows_reduce = series.profile_allows_reduce or series.category in reduce_ok
        if series.flexibility in _STOPPABLE and allows_stop:
            actions.append(
                SpendingAction(
                    kind=SpendingActionKind.STOP,
                    event_id=event_id,
                    new_amount=None,
                    series_event_ids=series.event_ids,
                    category=series.category,
                    normal_amount=normal,
                    series_key=key,
                )
            )
        if (
            series.flexibility in _REDUCIBLE
            and allows_reduce
            and series.minimum_allowed_amount is not None
        ):
            actions.append(
                SpendingAction(
                    kind=SpendingActionKind.REDUCE_TO,
                    event_id=event_id,
                    new_amount=series.minimum_allowed_amount,
                    series_event_ids=series.event_ids,
                    category=series.category,
                    normal_amount=normal,
                    series_key=key,
                )
            )
    return tuple(sorted(actions, key=lambda item: (item.event_id, item.kind.value)))


def action_combinations(
    actions: tuple[SpendingAction, ...],
    *,
    max_size: int = MAX_SPENDING_CHANGES,
) -> tuple[tuple[SpendingAction, ...], ...]:
    """Deterministic combinations of distinct series, size 1..max_size."""
    combos: list[tuple[SpendingAction, ...]] = []
    limit = min(max_size, len(actions))
    for size in range(1, limit + 1):
        for combo in combinations(actions, size):
            event_ids = [item.event_id for item in combo]
            series_keys = [item.series_key for item in combo]
            if len(event_ids) != len(set(event_ids)):
                continue
            if len(series_keys) != len(set(series_keys)):
                continue
            ordered = tuple(sorted(combo, key=lambda item: (item.event_id, item.kind.value)))
            combos.append(ordered)
    return tuple(combos)
