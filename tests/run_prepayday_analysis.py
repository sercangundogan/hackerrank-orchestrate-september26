"""Pre-payday obligation analysis. Not used by production."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from data.loader import load_dataset
from data.models import EventDirection, Flexibility
from data.repository import DatasetRepository
from decision.capacity import compute_capacity, infer_money_quantum, is_payment_safe
from finance.forecast_models import ForecastEventKind
from finance.models import CadenceConfidence
from tests.reference_reconstruction import load_resolved_samples, production_config

_FLEX = {
    Flexibility.REDUCIBLE,
    Flexibility.STOPPABLE,
    Flexibility.REDUCIBLE_OR_STOPPABLE,
}
_VARIABLE = {"groceries", "transport", "utilities", "healthcare"}
_ZERO = Decimal("0")


def _day(event):
    return event.cash_date or event.event_date


def first_income_date(forecast):
    dates = [
        entry.date
        for entry in forecast.all_entries
        if entry.signed_amount > 0
    ]
    return min(dates) if dates else None


def trough_date(forecast):
    best = None
    best_min = None
    for day in forecast.daily_forecasts:
        if best_min is None or day.minimum_balance < best_min:
            best_min = day.minimum_balance
            best = day.date
    return best, best_min


def outflows_before(forecast, end: date | None):
    rows = []
    total = _ZERO
    cats = Counter()
    for entry in forecast.all_entries:
        if entry.signed_amount >= 0:
            continue
        if end is not None and entry.date > end:
            continue
        rows.append(entry)
        total += entry.amount_home_currency
        cats[entry.category or "?"] += entry.amount_home_currency
    return total, cats, rows


def historical_salary_dates(state):
    dates = []
    for event in state.historical_settled_cash_events:
        day = _day(event)
        if day is None or day >= state.request_date:
            continue
        if event.category == "salary" and event.direction is EventDirection.CREDIT:
            dates.append(day)
    return sorted(set(dates))


def hist_debits_before_paydays(state, lookback_cycles=3):
    salaries = historical_salary_dates(state)
    if len(salaries) < 2:
        return []
    cycles = list(zip(salaries, salaries[1:]))[-lookback_cycles:]
    buckets = []
    for start, end in cycles:
        days = (end - start).days or 1
        items = []
        for event in state.historical_settled_cash_events:
            day = _day(event)
            if day is None or event.direction is not EventDirection.DEBIT:
                continue
            if not (start <= day < end):
                continue
            if event.amount_home_currency is None:
                continue
            rel = (day - end).days
            items.append((rel, event.category, event.amount_home_currency, event.description, event.flexibility))
        buckets.append((start, end, days, items))
    return buckets


def omitted_series_before(state, forecast, end: date | None):
    projected = {
        (entry.category, (entry.description or "").lower())
        for entry in forecast.all_entries
        if entry.signed_amount < 0 and (end is None or entry.date <= end)
    }
    omitted = []
    for series in state.recurring_series_candidates:
        if series.direction is not EventDirection.DEBIT:
            continue
        key = (series.category, series.original_description.lower())
        if any(series.category == cat and series.original_description.lower() in desc for cat, desc in projected):
            continue
        last = series.observed_dates[-1] if series.observed_dates else None
        omitted.append(
            (
                series.category,
                series.original_description,
                series.inferred_cadence.value,
                series.cadence_confidence.value,
                len(series.event_ids),
                series.flexibility.value if series.flexibility else None,
                series.likely_recurring,
                last,
                series.representative_amount,
            )
        )
    return omitted


def main() -> None:
    repo = DatasetRepository(load_dataset())
    cfg = production_config(repo)
    print("=== FIRST-TROUGH / PRE-PAYDAY ===")
    print(
        "id\treq\tfirst_inc\ttrough\tlabel\tpred\tdelta\tpre_out\tfloor\t"
        "pathmin\thead\tlabel_head_gap"
    )
    for sample, state, _ in load_resolved_samples(repo):
        cap = compute_capacity(state, sample.request, cfg)
        inc = first_income_date(cap.baseline)
        trough, tmin = trough_date(cap.baseline)
        limit = inc
        if trough is not None and (limit is None or trough < limit):
            limit = trough
        pre_out, cats, rows = outflows_before(cap.baseline, inc - timedelta(days=1) if inc else None)
        label = sample.decision.amount_safe_to_pay
        pred = cap.amount_safe_to_pay
        floor = state.profile.minimum_balance_to_keep
        # path min before income
        mins = []
        for day in cap.baseline.daily_forecasts:
            if inc and day.date >= inc:
                break
            mins.append(day.minimum_balance)
        pathmin = min(mins) if mins else cap.baseline.minimum_observed_balance
        head = pathmin - floor
        print(
            f"{sample.request.request_id}\t{sample.request.request_date}\t{inc}\t{trough}\t"
            f"{label}\t{pred}\t{pred-label}\t{pre_out}\t{floor}\t{pathmin}\t{head}\t{head-label}"
        )
        print("   pre_cats", dict(cats))
        print(
            "   status",
            sample.decision.affordability_status.value,
            "earliest_l",
            sample.decision.earliest_date_for_full_payment,
            "earliest_p",
            cap.earliest_date_for_full_payment,
        )
        omitted = omitted_series_before(state, cap.baseline, inc - timedelta(days=1) if inc else trough)
        flex_omitted = [row for row in omitted if row[5] in {f.value for f in _FLEX} and row[3] in {"high", "medium"}]
        low_omitted = [row for row in omitted if row[3] == "low" and row[6]]
        print("   HIGH/MED flex omitted", [(r[0], r[1], r[3], r[4], r[8]) for r in flex_omitted][:8])
        print("   LOW likely omitted", [(r[0], r[1], r[2], r[4], r[7], r[8]) for r in low_omitted if r[0] in {"dining","entertainment","shopping","streaming","gym","cloud_storage"}][:10])

    print("\n=== RELATIVE-TO-PAYDAY HISTORICAL DEBITS (last 3 cycles) ===")
    for sample, state, _ in load_resolved_samples(repo):
        cycles = hist_debits_before_paydays(state)
        if not cycles:
            print(sample.request.request_id, "no salary cycles")
            continue
        rel = defaultdict(list)
        for _s, _e, _d, items in cycles:
            for offset, cat, amt, desc, flex in items:
                if cat in _VARIABLE or cat in {"rent", "dining", "shopping", "entertainment"}:
                    rel[cat].append((offset, amt))
        print(sample.request.request_id, "cycles", len(cycles), "last_len", cycles[-1][2])
        for cat, rows in sorted(rel.items()):
            offs = [o for o, _ in rows]
            if not offs:
                continue
            print(
                f"   {cat} n={len(offs)} med_rel={median(offs)} "
                f"p25={sorted(offs)[len(offs)//4]} p75={sorted(offs)[3*len(offs)//4]} "
                f"sum={sum((a for _,a in rows), _ZERO)}"
            )

    print("\n=== REQUEST 05 / 11 / 13 DEEP ===")
    for rid in ("request_05", "request_11", "request_13"):
        for sample, state, _ in load_resolved_samples(repo):
            if sample.request.request_id != rid:
                continue
            cap = compute_capacity(state, sample.request, cfg)
            inc = first_income_date(cap.baseline)
            print("\n", rid, "opening", state.opening_balance, "floor", state.profile.minimum_balance_to_keep)
            print(" protect", state.profile.expense_categories_to_protect)
            print(" reduce", state.profile.expense_categories_user_is_willing_to_reduce)
            print(" stop", state.profile.expense_categories_user_is_willing_to_stop)
            print(" pending", [(e.category, e.amount_home_currency, e.description) for e in state.pending_debits])
            print(" scheduled_debits", [(e.category, e.amount_home_currency, e.cash_date or e.event_date, e.description) for e in state.scheduled_debits])
            print(" label", sample.decision.amount_safe_to_pay, "pred", cap.amount_safe_to_pay)
            print(" expl", sample.decision.decision_explanation[:280])
            print(" changes", sample.decision.spending_changes_needed)
            # all forecast debits before income
            end = inc - timedelta(days=1) if inc else cap.baseline.horizon_end
            print(" first income", inc)
            for e in cap.baseline.all_entries:
                if e.date > end:
                    continue
                print("  ", e.date, e.signed_amount, e.category, e.kind.value, e.description[:70])
            cycles = hist_debits_before_paydays(state, lookback_cycles=3)
            for start, endc, days, items in cycles:
                total = sum((a for *_, a, _d, _f in [(i[0], i[1], i[2], i[3], i[4]) for i in items]), _ZERO)
                byc = Counter()
                for _rel, cat, amt, _d, _f in items:
                    byc[cat] += amt
                print("  hist cycle", start, endc, "days", days, "total", total, dict(byc))
            if rid == "request_13":
                for s in state.recurring_series_candidates:
                    if s.category == "salary":
                        print("  income series", s.original_description, s.cadence_confidence, s.scheduled_confirmed_event_ids, s.observed_dates[-3:])

    print("\n=== LABEL vs HEADROOM DISTRIBUTION ===")
    kinds = Counter()
    for sample, state, _ in load_resolved_samples(repo):
        cap = compute_capacity(state, sample.request, cfg)
        label = sample.decision.amount_safe_to_pay
        floor = state.profile.minimum_balance_to_keep
        inc = first_income_date(cap.baseline)
        mins = []
        for day in cap.baseline.daily_forecasts:
            if inc and day.date >= inc:
                break
            mins.append(day.minimum_balance)
        pathmin = min(mins) if mins else cap.baseline.minimum_observed_balance
        head = pathmin - floor
        paid = is_payment_safe(state, sample.request, sample.request.request_date, label, cfg)
        q = infer_money_quantum(sample.request.requested_amount, state.opening_balance, floor)
        near_floor = abs(paid.minimum_observed_balance - floor) <= q
        near_head = abs(head - label) <= q
        capped = label == sample.request.requested_amount
        if near_floor:
            kinds["label_hits_floor"] += 1
        elif near_head:
            kinds["label_eq_prepayday_head"] += 1
        elif capped and pred_full(cap, sample):
            kinds["both_full_requested"] += 1
        else:
            kinds["other"] += 1
            print(
                " other",
                sample.request.request_id,
                "label",
                label,
                "head",
                head,
                "pred",
                cap.amount_safe_to_pay,
                "floor_gap_if_label",
                paid.minimum_observed_balance - floor,
            )
    print(dict(kinds))


def pred_full(cap, sample):
    return cap.amount_safe_to_pay == sample.request.requested_amount


if __name__ == "__main__":
    main()
