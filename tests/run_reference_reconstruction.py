"""Run reference-rule reconstruction analysis. Not used by production."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from statistics import median

from data.loader import load_dataset
from data.models import EventDirection
from data.repository import DatasetRepository
from decision.capacity import compute_capacity, infer_money_quantum, is_payment_safe
from finance.essential_spending import essential_spend_entries, forecast_config_from_profiles
from finance.forecast import forecast_financial_state
from finance.forecast_models import EssentialSpendStrategy, ForecastEventKind
from tests.capacity_evaluation import evaluate_sample_capacity
from tests.reference_models import (
    EXPERIMENTAL_BUILDERS,
    _CONTRACTUAL,
    _VARIABLE,
    summarize_capacity_rows,
)
from tests.reference_reconstruction import (
    implied_gaps,
    load_resolved_samples,
    production_config,
    salary_audit,
)

import finance.forecast as forecast_mod


def _floor_hits(repository: DatasetRepository, config) -> int:
    hits = 0
    for sample, state, _bundle in load_resolved_samples(repository):
        label = sample.decision.amount_safe_to_pay
        paid = is_payment_safe(state, sample.request, sample.request.request_date, label, config)
        quantum = infer_money_quantum(
            sample.request.requested_amount,
            state.opening_balance,
            state.profile.minimum_balance_to_keep,
        )
        if abs(paid.minimum_observed_balance - state.profile.minimum_balance_to_keep) <= quantum:
            hits += 1
    return hits


def _evaluate(name: str, repository: DatasetRepository, config, builder=None):
    original = forecast_mod.essential_spend_entries
    try:
        if builder is None:
            forecast_mod.essential_spend_entries = essential_spend_entries
        else:
            forecast_mod.essential_spend_entries = builder
        rows = evaluate_sample_capacity(repository, config)
        floors = _floor_hits(repository, config)
        return summarize_capacity_rows(name, rows, floors), rows
    finally:
        forecast_mod.essential_spend_entries = original


def _print_metrics(metrics) -> None:
    print(
        f"{metrics.name:24} amt={metrics.amount_exact:2}/25  "
        f"date={metrics.earliest_exact:2}/25  "
        f"MAE={metrics.mae:.2f}  "
        f"med%={float(metrics.median_norm)*100:.2f}  "
        f"w1={metrics.within_1} w5={metrics.within_5} w10={metrics.within_10}  "
        f"over={metrics.over} under={metrics.under}  "
        f"floor={metrics.floor_hits}/25"
    )


def abc_breakdown(repository: DatasetRepository, config) -> None:
    print("\n=== A/B/C 90-day signed sums vs label (production forecast) ===")
    print("id\topen\tfloor\tA_fixed\tB_var\tC_inc\tstatic_head\tlabel\tdelta_static\tpred")
    for sample, state, _bundle in load_resolved_samples(repository):
        cap = compute_capacity(state, sample.request, config)
        a = b = c = Decimal("0")
        for entry in cap.baseline.all_entries:
            if entry.kind is ForecastEventKind.CANDIDATE_PAYMENT:
                continue
            if entry.signed_amount > 0:
                c += entry.amount_home_currency
            elif (entry.category or "") in _CONTRACTUAL:
                a += entry.amount_home_currency
            elif (entry.category or "") in _VARIABLE:
                b += entry.amount_home_currency
            else:
                a += entry.amount_home_currency
        static = state.opening_balance - state.profile.minimum_balance_to_keep - a - b + c
        label = sample.decision.amount_safe_to_pay
        print(
            f"{sample.request.request_id}\t{state.opening_balance}\t"
            f"{state.profile.minimum_balance_to_keep}\t{a}\t{b}\t{c}\t"
            f"{static}\t{label}\t{static-label}\t{cap.amount_safe_to_pay}"
        )


def salary_detail(repository: DatasetRepository, config) -> None:
    print("\n=== SALARY / CREDIT AUDIT ===")
    for sample, state, _bundle in load_resolved_samples(repository):
        forecast = forecast_financial_state(state, config)
        audit = salary_audit(state, forecast)
        credits = [
            entry
            for entry in forecast.all_entries
            if entry.signed_amount > 0
        ]
        lines = []
        for entry in credits:
            tag = "GEN" if entry.is_generated_recurrence else "EXP"
            lines.append(
                f"{tag}:{entry.date}:{entry.category}:{entry.amount_home_currency}:{entry.description[:40]}"
            )
        print(
            sample.request.request_id,
            "n",
            len(credits),
            "adj",
            audit["adjustments"],
            "non_salary",
            audit["non_salary_credits"],
        )
        for line in lines:
            print("   ", line)


def precision_cases(repository: DatasetRepository, config) -> None:
    print("\n=== NEAR-MISS PRECISION ===")
    targets = {
        "request_06",
        "request_21",
        "request_18",
        "request_22",
        "request_24",
        "request_09",
        "request_15",
    }
    for sample, state, _bundle in load_resolved_samples(repository):
        if sample.request.request_id not in targets:
            continue
        cap = compute_capacity(state, sample.request, config)
        label = sample.decision.amount_safe_to_pay
        paid = is_payment_safe(state, sample.request, sample.request.request_date, label, config)
        quantum = cap.quantum
        print(
            sample.request.request_id,
            "pred",
            cap.amount_safe_to_pay,
            "label",
            label,
            "delta",
            cap.amount_safe_to_pay - label,
            "q",
            quantum,
            "label_min",
            paid.minimum_observed_balance,
            "floor",
            state.profile.minimum_balance_to_keep,
            "unsafe_inc_min",
            cap.unsafe_increment_minimum,
            "requested",
            sample.request.requested_amount,
        )
        reserves = [
            (e.date, e.category, e.amount_home_currency)
            for e in cap.baseline.all_entries
            if e.kind is ForecastEventKind.ESSENTIAL_SPEND_RESERVE
        ]
        print("   reserves", reserves[:8], "n", len(reserves), "sum", sum((a for _,_,a in reserves), Decimal(0)))


def until_next_salary(repository: DatasetRepository, config) -> None:
    print("\n=== PROTECT-UNTIL-NEXT-SALARY STATIC ===")
    print("id\tnext_sal\tdays\tA\tBhist_frac\thead\tlabel\tpred")
    for sample, state, _bundle in load_resolved_samples(repository):
        cap = compute_capacity(state, sample.request, config)
        next_sal = None
        for entry in cap.baseline.all_entries:
            if entry.signed_amount > 0 and entry.category == "salary":
                next_sal = entry.date
                break
        if next_sal is None:
            print(sample.request.request_id, "no future salary")
            continue
        days = (next_sal - sample.request.request_date).days
        a = b = Decimal("0")
        for entry in cap.baseline.all_entries:
            if entry.date > next_sal:
                continue
            if entry.kind is ForecastEventKind.CANDIDATE_PAYMENT:
                continue
            if entry.signed_amount > 0:
                continue
            if (entry.category or "") in _VARIABLE:
                b += entry.amount_home_currency
            else:
                a += entry.amount_home_currency
        head = state.opening_balance - state.profile.minimum_balance_to_keep - a - b
        print(
            f"{sample.request.request_id}\t{next_sal}\t{days}\t{a}\t{b}\t{head}\t"
            f"{sample.decision.amount_safe_to_pay}\t{cap.amount_safe_to_pay}"
        )


def main() -> None:
    repository = DatasetRepository(load_dataset())
    config = production_config(repository)
    print("=== IMPLIED GAPS (production daily_rate) ===")
    gaps = implied_gaps(repository, config)
    extra = sum(1 for row in gaps if row.implied_forecast_gap > 0)
    tight = sum(1 for row in gaps if row.implied_forecast_gap < 0)
    print(f"extra_slack={extra} too_tight={tight} near_floor={sum(1 for r in gaps if r.near_floor)}")
    for row in gaps:
        print(
            f"{row.request_id}\tpred={row.predicted}\tlabel={row.labeled}\t"
            f"delta={row.delta}\timplied_gap={row.implied_forecast_gap}\t"
            f"safe={row.is_label_safe}\tlim={row.limiting_date}\t{row.limiting_event}"
        )

    print("\n=== CANDIDATE MODELS ===")
    metrics, prod_rows = _evaluate("prod_daily_rate", repository, config, builder=None)
    _print_metrics(metrics)
    all_metrics = [metrics]
    for name, builder in EXPERIMENTAL_BUILDERS.items():
        metrics, _rows = _evaluate(name, repository, config, builder=builder)
        _print_metrics(metrics)
        all_metrics.append(metrics)

    abc_breakdown(repository, config)
    until_next_salary(repository, config)
    salary_detail(repository, config)
    precision_cases(repository, config)

    print("\n=== MODEL TABLE (copy) ===")
    print("model,amount_exact,earliest_exact,MAE,median_norm,w1,w5,w10,over,under,floor")
    for item in all_metrics:
        print(
            f"{item.name},{item.amount_exact},{item.earliest_exact},{item.mae},"
            f"{item.median_norm},{item.within_1},{item.within_5},{item.within_10},"
            f"{item.over},{item.under},{item.floor_hits}"
        )


if __name__ == "__main__":
    main()
