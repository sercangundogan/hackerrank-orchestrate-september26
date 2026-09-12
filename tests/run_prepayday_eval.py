"""Compare pre-payday timing models. Analysis only."""

from __future__ import annotations

from finance.essential_spending import essential_spend_entries, forecast_config_from_profiles
from finance.forecast_models import EssentialSpendStrategy
from data.loader import load_dataset
from data.repository import DatasetRepository
from tests.capacity_evaluation import evaluate_sample_capacity
from tests.reference_models import (
    EXPERIMENTAL_BUILDERS,
    summarize_capacity_rows,
)
import finance.forecast as forecast_mod


def main() -> None:
    repo = DatasetRepository(load_dataset())
    config = forecast_config_from_profiles(
        repo.dataset.profiles,
        essential_spend_strategy=EssentialSpendStrategy.DAILY_RATE,
        essential_lookback_days=90,
        strict_unresolved_amounts=True,
    )
    baseline = evaluate_sample_capacity(repo, config)
    base_err = {row.request_id: row.amount_error for row in baseline}
    print("prod_weekly", summarize_capacity_rows("prod_weekly", baseline, 0))
    original = forecast_mod.essential_spend_entries
    names = (
        "G_hist90_weekly",
        "D_cadence_mean",
        "prepayday_frontload",
        "relative_payday",
    )
    try:
        for name in names:
            forecast_mod.essential_spend_entries = EXPERIMENTAL_BUILDERS[name]
            rows = evaluate_sample_capacity(repo, config)
            metrics = summarize_capacity_rows(name, rows, 0)
            improved = [
                row.request_id
                for row in rows
                if abs(row.amount_error) < abs(base_err[row.request_id])
            ]
            worsened = [
                row.request_id
                for row in rows
                if abs(row.amount_error) > abs(base_err[row.request_id])
            ]
            print(
                f"{name:22} amt={metrics.amount_exact} date={metrics.earliest_exact} "
                f"MAE={metrics.mae:.2f} med%={float(metrics.median_norm)*100:.2f} "
                f"w1={metrics.within_1} w5={metrics.within_5} over={metrics.over} under={metrics.under}"
            )
            print("  improved", improved)
            print("  worsened", worsened)
            focus = {row.request_id: row for row in rows}
            for rid in ("request_05", "request_11", "request_13", "request_01", "request_06"):
                row = focus[rid]
                print(f"  {rid} pred={row.predicted_amount} label={row.labeled_amount} err={row.amount_error}")
    finally:
        forecast_mod.essential_spend_entries = original


if __name__ == "__main__":
    main()
