"""Evaluate future-income policies. Not used by production."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal

from ai.client import ModelClient
from data.loader import load_dataset
from data.repository import DatasetRepository
from evidence.cache import EvidenceCache
from evidence.pipeline import resolve_request_state
from finance.essential_spending import forecast_config_from_profiles
from finance.forecast import forecast_financial_state
from finance.forecast_models import EssentialSpendStrategy, SalaryProjectionMode
from tests.capacity_evaluation import evaluate_sample_capacity
from tests.reference_models import summarize_capacity_rows
from tests.reference_reconstruction import load_resolved_samples


FOCUS = {
    "request_02",
    "request_05",
    "request_07",
    "request_11",
    "request_12",
    "request_13",
}


def _config(mode: SalaryProjectionMode):
    return forecast_config_from_profiles(
        DatasetRepository(load_dataset()).dataset.profiles,
        essential_spend_strategy=EssentialSpendStrategy.DAILY_RATE,
        essential_lookback_days=90,
        strict_unresolved_amounts=True,
        salary_projection_mode=mode,
    )


def _metrics(name, rows, baseline_dates=None):
    metrics = summarize_capacity_rows(name, rows, 0)
    later = earlier = unchanged = 0
    if baseline_dates:
        for row in rows:
            old = baseline_dates.get(row.request_id)
            new = row.predicted_earliest
            if old == new:
                unchanged += 1
            elif old is None:
                earlier += 1
            elif new is None:
                later += 1
            elif new > old:
                later += 1
            else:
                earlier += 1
    return metrics, later, earlier, unchanged


def _print(metrics, later=None, earlier=None, unchanged=None) -> None:
    extra = ""
    if later is not None:
        extra = f"  earliest_later={later} earlier={earlier} unchanged={unchanged}"
    print(
        f"{metrics.name:28} amt={metrics.amount_exact:2}/25 date={metrics.earliest_exact:2}/25 "
        f"MAE={metrics.mae:.2f} med%={float(metrics.median_norm)*100:.2f} "
        f"w1={metrics.within_1} w5={metrics.within_5} w10={metrics.within_10} "
        f"over={metrics.over} under={metrics.under}{extra}"
    )


def full_dataset_impact(repository: DatasetRepository, mode: SalaryProjectionMode) -> None:
    cfg = forecast_config_from_profiles(
        repository.dataset.profiles,
        essential_spend_strategy=EssentialSpendStrategy.DAILY_RATE,
        essential_lookback_days=90,
        strict_unresolved_amounts=True,
        salary_projection_mode=mode,
    )
    cache = EvidenceCache()
    client = ModelClient(api_key="unused")
    n_gen = n_exp = zero = hist_only = confirmed = 0
    by_ccy = defaultdict(lambda: Decimal("0"))
    subtype = Counter()
    for request in repository.dataset.evaluation_requests:
        _, state, _ = resolve_request_state(repository, request, client=client, cache=cache)
        result = forecast_financial_state(state, cfg)
        credits = [entry for entry in result.all_entries if entry.signed_amount > 0]
        if not credits:
            zero += 1
        if any(entry.message_confirmed or entry.explicitly_scheduled for entry in credits):
            confirmed += 1
        elif any(entry.generated_from_history for entry in credits):
            hist_only += 1
        for entry in credits:
            subtype[entry.income_subtype or "other"] += 1
            by_ccy[state.profile.home_currency.value] += entry.amount_home_currency
            if entry.is_generated_recurrence:
                n_gen += 1
            else:
                n_exp += 1
    print(
        f"  events gen={n_gen} explicit={n_exp} users_zero={zero} "
        f"history_only_users={hist_only} confirmed_users={confirmed}"
    )
    print("  subtype", dict(subtype))
    print("  totals", {k: str(v) for k, v in by_ccy.items()})


def main() -> None:
    repository = DatasetRepository(load_dataset())
    modes = (
        ("legacy_strong", SalaryProjectionMode.SCHEDULED_PLUS_REGULAR_HISTORY),
        ("A_strict", SalaryProjectionMode.STRICT_CONFIRMED),
        ("B_confirmed_continue", SalaryProjectionMode.CONFIRMED_THEN_CONTINUE),
        ("C_strong_base", SalaryProjectionMode.STRONG_BASE_SALARY),
    )
    baseline_rows = evaluate_sample_capacity(
        repository, _config(SalaryProjectionMode.SCHEDULED_PLUS_REGULAR_HISTORY)
    )
    baseline_dates = {row.request_id: row.predicted_earliest for row in baseline_rows}
    print("=== SAMPLE CAPACITY BY INCOME POLICY ===")
    for name, mode in modes:
        rows = evaluate_sample_capacity(repository, _config(mode))
        metrics, later, earlier, unchanged = _metrics(name, rows, baseline_dates)
        _print(metrics, later, earlier, unchanged)
        print("  focus")
        for row in rows:
            if row.request_id not in FOCUS:
                continue
            print(
                f"    {row.request_id} pred={row.predicted_amount} label={row.labeled_amount} "
                f"err={row.amount_error} earliest_p={row.predicted_earliest} "
                f"earliest_l={row.labeled_earliest}"
            )
    print("\n=== FULL DATASET IMPACT ===")
    for name, mode in modes:
        print(name)
        full_dataset_impact(repository, mode)

    print("\n=== SAMPLE CREDIT INVENTORY (production default mode) ===")
    cfg = _config(SalaryProjectionMode.SCHEDULED_PLUS_REGULAR_HISTORY)
    for sample, state, _ in load_resolved_samples(repository):
        result = forecast_financial_state(state, cfg)
        credits = [e for e in result.all_entries if e.signed_amount > 0]
        if not credits:
            print(sample.request.request_id, "NO_FUTURE_INCOME", [a.kind.value for a in state.forecast_adjustments])
            continue
        for entry in credits:
            print(
                f"{sample.request.request_id}\t{entry.date}\t{entry.amount_home_currency}\t"
                f"{entry.income_subtype}\t{entry.confirmation_type}\t"
                f"sched={entry.explicitly_scheduled}\tmsg={entry.message_confirmed}\t"
                f"hist={entry.generated_from_history}\t{entry.description[:70]}"
            )


if __name__ == "__main__":
    main()
