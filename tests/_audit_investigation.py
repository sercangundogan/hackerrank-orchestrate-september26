"""Isolated audit scripts. Does not change production modules."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ai.client import ModelClient
from data.loader import load_dataset
from data.models import EventDirection, EventStatus, Flexibility
from data.repository import DatasetRepository
from decision.capacity import compute_capacity
from decision.engine import decide
from decision.plans import format_payment_plan, format_spending_changes
from decision.spending_optimizer import (
    _FLEXIBLE,
    _REDUCIBLE,
    _STOPPABLE,
    latest_series_event_id,
    legal_actions,
    series_has_future_debit,
)
from evidence.cache import EvidenceCache
from evidence.pipeline import resolve_request_state
from finance.essential_spending import (
    build_category_profiles,
    eligible_essential_categories,
    forecast_config_from_profiles,
)
from finance.forecast import (
    _eligible_for_auto_projection,
    forecast_financial_state,
    next_recurrence_date,
    series_key,
)
from finance.forecast_models import ForecastEventKind
from finance.lifecycle import build_lifecycles
from finance.models import Cadence, CadenceConfidence, LifecycleType
from finance.recurrence import infer_cadence, normalize_description


OUT = Path("/tmp/buyorwait-audit")
OUT.mkdir(parents=True, exist_ok=True)


def _legal_without_forecast(series, state) -> bool:
    if series.direction is not EventDirection.DEBIT:
        return False
    if not series.event_ids:
        return False
    if series.flexibility not in _FLEXIBLE:
        return False
    protected = set(state.profile.expense_categories_to_protect)
    if series.is_protected_category or series.category in protected:
        return False
    reduce_ok = set(state.profile.expense_categories_user_is_willing_to_reduce)
    stop_ok = set(state.profile.expense_categories_user_is_willing_to_stop)
    allows_stop = series.profile_allows_stop or series.category in stop_ok
    allows_reduce = series.profile_allows_reduce or series.category in reduce_ok
    if series.flexibility in _STOPPABLE and allows_stop:
        return True
    if (
        series.flexibility in _REDUCIBLE
        and allows_reduce
        and series.minimum_allowed_amount is not None
    ):
        return True
    return False


def _reserved(series, forecast) -> Decimal:
    key = series_key(series)
    total = Decimal("0")
    for entry in forecast.all_entries:
        if entry.source_series_key == key and entry.signed_amount < 0:
            total += entry.amount_home_currency
    return total


def part1_request11(repo, cache, client, config):
    sample = next(s for s in repo.dataset.sample_requests if s.request.request_id == "request_11")
    request = sample.request
    _, state, _ = resolve_request_state(repo, request, client=client, cache=cache)
    forecast = forecast_financial_state(state, strategy_config=config)
    capacity = compute_capacity(state, request, config)
    result = decide(state, request, repo.payment_options_for_request(request.request_id), config, capacity=capacity)

    dining_events = [
        e for e in state.classified_events
        if e.category == "dining" and e.direction is EventDirection.DEBIT
    ]
    dining_events.sort(key=lambda e: e.event_date)

    series_rows = []
    for series in state.recurring_series_candidates:
        if series.category != "dining":
            continue
        series_rows.append({
            "event_ids": list(series.event_ids),
            "desc": series.normalized_description,
            "flex": series.flexibility.value,
            "cadence": series.inferred_cadence.value,
            "confidence": series.cadence_confidence.value,
            "obs": len(series.observed_dates),
            "dates": [d.isoformat() for d in series.observed_dates],
            "amounts": [str(a) for a in series.observed_amounts_home_currency],
            "min_allowed": str(series.minimum_allowed_amount),
            "likely": series.likely_recurring,
            "auto_proj": _eligible_for_auto_projection(series, config),
            "in_forecast": series_has_future_debit(series, forecast),
            "baseline_reserved": str(_reserved(series, forecast)),
            "legal_wo_forecast": _legal_without_forecast(series, state),
            "prod_optimizer": any(
                a.event_id == latest_series_event_id(series) for a in legal_actions(state, forecast)
            ),
        })

    dates = tuple(e.event_date for e in dining_events if e.event_date)
    cad = infer_cadence(dates) if dates else None
    latest = dining_events[-1] if dining_events else None

    flexible = [s for s in state.recurring_series_candidates if s.direction is EventDirection.DEBIT and s.flexibility.value != "fixed"]
    forecasted_flex = [
        {
            "latest": latest_series_event_id(s),
            "cat": s.category,
            "desc": s.normalized_description,
            "reserved": str(_reserved(s, forecast)),
            "cadence": s.inferred_cadence.value,
            "conf": s.cadence_confidence.value,
        }
        for s in flexible
        if series_has_future_debit(s, forecast)
    ]

    payload = {
        "request_date": request.request_date.isoformat(),
        "deadline": request.desired_completion_date.isoformat(),
        "requested": str(request.requested_amount),
        "profile": {
            "protect": list(state.profile.expense_categories_to_protect),
            "reduce": list(state.profile.expense_categories_user_is_willing_to_reduce),
            "stop": list(state.profile.expense_categories_user_is_willing_to_stop),
            "min_balance": str(state.profile.minimum_balance_to_keep),
            "balance": str(state.profile.current_available_balance),
            "max_installment_months": state.profile.max_installment_months,
            "methods": [m.value for m in state.profile.payment_methods_user_will_consider],
        },
        "label": {
            "amount": str(sample.decision.amount_safe_to_pay),
            "status": sample.decision.affordability_status.value,
            "method": sample.decision.recommended_payment_method.value,
            "plan": sample.decision.payment_plan,
            "earliest": sample.decision.earliest_date_for_full_payment.isoformat()
            if sample.decision.earliest_date_for_full_payment else "",
            "changes": sample.decision.spending_changes_needed,
        },
        "predicted": {
            "amount": str(result.amount_safe_to_pay),
            "status": result.affordability_status.value,
            "method": result.recommended_payment_method.value,
            "plan": format_payment_plan(result.payment_plan),
            "earliest": result.earliest_date_for_full_payment.isoformat()
            if result.earliest_date_for_full_payment else "",
            "changes": format_spending_changes(result.spending_changes_needed),
            "full_safe_today": capacity.full_payment_safe_today,
        },
        "dining_events": [
            {
                "id": e.source_event_id,
                "desc": e.description,
                "date": e.event_date.isoformat() if e.event_date else None,
                "amount": str(e.amount_home_currency),
                "flex": e.flexibility.value,
                "min": str(e.minimum_allowed_amount),
                "status": e.status.value,
            }
            for e in dining_events
        ],
        "combined_dining_cadence": {
            "n": len(dates),
            "cadence": cad[0].value if cad else None,
            "days": cad[1] if cad else None,
            "confidence": cad[2].value if cad else None,
            "reason": cad[3] if cad else None,
            "latest_id": latest.source_event_id if latest else None,
            "next_if_21d": (latest.event_date + timedelta(days=21)).isoformat() if latest and latest.event_date else None,
        },
        "description_series": series_rows,
        "forecasted_flexible": forecasted_flex,
        "legal_actions_prod": [f"{a.kind.value}:{a.event_id}" for a in legal_actions(state, forecast)],
        "essential_eligible": sorted(eligible_essential_categories(state.profile, config.extra_essential_categories)),
        "essential_profiles": [
            {
                "cat": p.category,
                "residual": str(p.residual_reserve),
                "expected": str(p.projected_budget),
                "explicit": str(p.explicit_projected_amount),
                "n_hist": len(p.historical_event_ids),
            }
            for p in build_category_profiles(state, config, forecast.all_entries)
            if p.residual_reserve > 0
        ],
    }
    (OUT / "part1_request11.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _category_groups(state):
    groups = defaultdict(list)
    for series in state.recurring_series_candidates:
        if series.direction is not EventDirection.DEBIT:
            continue
        groups[(series.category, series.flexibility)].append(series)
    return groups


def part1_scan(repo, cache, client, config):
    rows = []
    category_recon = []
    requests = [s.request for s in repo.dataset.sample_requests]
    requests.extend(repo.dataset.evaluation_requests)
    by_user_state = {}
    for request in requests:
        _, state, _ = resolve_request_state(repo, request, client=client, cache=cache)
        forecast = forecast_financial_state(state, strategy_config=config)
        by_user_state[request.request_id] = (request, state, forecast)
        for series in state.recurring_series_candidates:
            if not _legal_without_forecast(series, state):
                continue
            reserved = _reserved(series, forecast)
            prod = series_has_future_debit(series, forecast)
            rows.append({
                "request_id": request.request_id,
                "eval": request.request_id not in {s.request.request_id for s in repo.dataset.sample_requests}
                    and request.request_id.startswith("request_"),
                "is_sample": any(s.request.request_id == request.request_id for s in repo.dataset.sample_requests),
                "event_id": latest_series_event_id(series),
                "user_id": series.user_id,
                "series": f"{series.category}/{series.normalized_description}",
                "category": series.category,
                "flexibility": series.flexibility.value,
                "recurrence": series.inferred_cadence.value,
                "confidence": series.cadence_confidence.value,
                "observation_count": len(series.observed_dates),
                "likely_recurring": series.likely_recurring,
                "auto_proj": _eligible_for_auto_projection(series, config),
                "baseline_reserved_amount": str(reserved),
                "optimizer_change_allowed": prod,
                "preference_legal": True,
                "inconsistency": (not prod) or reserved == 0,
            })
        groups = _category_groups(state)
        for (category, flex), members in groups.items():
            if flex not in _FLEXIBLE:
                continue
            if len(members) < 2:
                continue
            dates = []
            amounts = []
            event_ids = []
            for member in members:
                dates.extend(member.observed_dates)
                amounts.extend(member.observed_amounts_home_currency)
                event_ids.extend(member.event_ids)
            paired = sorted(zip(dates, amounts, event_ids), key=lambda x: x[0])
            if len(paired) < 3:
                continue
            ordered_dates = tuple(p[0] for p in paired)
            cad, days, conf, reason = infer_cadence(ordered_dates)
            gaps = [(b - a).days for a, b in zip(ordered_dates, ordered_dates[1:])]
            spread = (max(gaps) - min(gaps)) if gaps else None
            stable_unnamed = (
                len(ordered_dates) >= 4
                and spread is not None
                and spread <= 3
                and days is not None
                and 5 <= days <= 40
            )
            named = cad in {Cadence.WEEKLY, Cadence.BIWEEKLY, Cadence.MONTHLY} and conf is not CadenceConfidence.LOW
            any_auto = any(_eligible_for_auto_projection(m, config) for m in members)
            legal = any(_legal_without_forecast(m, state) for m in members)
            if not legal:
                continue
            if any_auto:
                continue
            if not (named or stable_unnamed):
                continue
            latest_id = paired[-1][2]
            category_recon.append({
                "request_id": request.request_id,
                "user_id": state.profile.user_id,
                "is_sample": any(s.request.request_id == request.request_id for s in repo.dataset.sample_requests),
                "category": category,
                "flexibility": flex.value,
                "n_desc_series": len(members),
                "observation_count": len(paired),
                "cadence": cad.value,
                "cadence_days": days,
                "spread": spread,
                "stable_unnamed": stable_unnamed and not named,
                "confidence": conf.value,
                "reason": reason,
                "latest_event_id": latest_id,
                "descriptions": [m.normalized_description for m in members],
                "min_allowed": str(members[-1].minimum_allowed_amount),
                "any_desc_in_forecast": any(series_has_future_debit(m, forecast) for m in members),
            })
    (OUT / "part1_legal_flexible.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "part1_category_recon.json").write_text(json.dumps(category_recon, indent=2), encoding="utf-8")
    return rows, category_recon, by_user_state


def part2_duplicates(repo, cache, client, config, by_user_state):
    events = repo.dataset.events
    events_by_id = repo.dataset.events_by_id
    groups = [g for g in build_lifecycles(events, events_by_id)
              if g.lifecycle_type is LifecycleType.POSSIBLE_DUPLICATE_PENDING_CHARGE]
    messages = repo.dataset.messages
    images = repo.dataset.images
    requests = list(repo.dataset.evaluation_requests) + [s.request for s in repo.dataset.sample_requests]

    table = []
    impact = []
    for group in groups:
        parent = events_by_id[group.parent_event_id]
        child = events_by_id[group.child_event_id]
        user_id = child.user_id
        related_msgs = [
            {"id": m.message_id, "related": m.related_event_id, "text": m.message_text[:240]}
            for m in messages
            if m.user_id == user_id and (
                m.related_event_id in {parent.event_id, child.event_id} or "duplicate" in (m.body or "").lower()
            )
        ]
        related_imgs = [
            {"id": img.image_id, "related": img.related_event_id}
            for img in images
            if img.user_id == user_id and img.related_event_id in {parent.event_id, child.event_id}
        ]
        user_requests = [r for r in requests if r.user_id == user_id]
        reserved_on = []
        for request in user_requests:
            _, state, _ = resolve_request_state(repo, request, client=client, cache=cache)
            pending_ids = {e.source_event_id for e in state.pending_debits}
            reserved_on.append({
                "request_id": request.request_id,
                "request_date": request.request_date.isoformat(),
                "child_reserved": child.event_id in pending_ids,
                "pending_amount": str(next((e.amount_home_currency for e in state.pending_debits if e.source_event_id == child.event_id), None)),
            })
            if child.event_id in pending_ids:
                # counterfactual: drop this pending debit
                new_pending = tuple(e for e in state.pending_debits if e.source_event_id != child.event_id)
                new_ignored = state.ignored_events + tuple(
                    e for e in state.pending_debits if e.source_event_id == child.event_id
                )
                alt_state = replace(state, pending_debits=new_pending, ignored_events=new_ignored)
                options = repo.payment_options_for_request(request.request_id)
                base_cap = compute_capacity(state, request, config)
                alt_cap = compute_capacity(alt_state, request, config)
                base_dec = decide(state, request, options, config, capacity=base_cap)
                alt_dec = decide(alt_state, request, options, config, capacity=alt_cap)
                impact.append({
                    "request_id": request.request_id,
                    "user_id": user_id,
                    "child": child.event_id,
                    "parent": parent.event_id,
                    "pending_amount": str(child.amount),
                    "base_amount": str(base_dec.amount_safe_to_pay),
                    "alt_amount": str(alt_dec.amount_safe_to_pay),
                    "amount_delta": str(alt_dec.amount_safe_to_pay - base_dec.amount_safe_to_pay),
                    "base_status": base_dec.affordability_status.value,
                    "alt_status": alt_dec.affordability_status.value,
                    "base_method": base_dec.recommended_payment_method.value,
                    "alt_method": alt_dec.recommended_payment_method.value,
                    "base_earliest": base_dec.earliest_date_for_full_payment.isoformat() if base_dec.earliest_date_for_full_payment else "",
                    "alt_earliest": alt_dec.earliest_date_for_full_payment.isoformat() if alt_dec.earliest_date_for_full_payment else "",
                    "changed": (
                        base_dec.amount_safe_to_pay != alt_dec.amount_safe_to_pay
                        or base_dec.affordability_status != alt_dec.affordability_status
                        or base_dec.recommended_payment_method != alt_dec.recommended_payment_method
                        or base_dec.earliest_date_for_full_payment != alt_dec.earliest_date_for_full_payment
                    ),
                })
        same_amount = parent.amount == child.amount
        same_currency = parent.currency == child.currency
        table.append({
            "parent": parent.event_id,
            "child": child.event_id,
            "user_id": user_id,
            "linked_event_id": child.linked_event_id,
            "parent_date": parent.event_date.isoformat(),
            "parent_settle": parent.settlement_date.isoformat() if parent.settlement_date else None,
            "child_date": child.event_date.isoformat(),
            "child_settle": child.settlement_date.isoformat() if child.settlement_date else None,
            "amount": str(child.amount),
            "currency": child.currency.value,
            "parent_desc": parent.description,
            "child_desc": child.description,
            "parent_status": parent.status.value,
            "child_status": child.status.value,
            "category": child.category,
            "same_amount": same_amount,
            "same_currency": same_currency,
            "messages": related_msgs,
            "images": related_imgs,
            "reserved_on_requests": reserved_on,
        })
    (OUT / "part2_duplicates.json").write_text(json.dumps({"table": table, "impact": impact}, indent=2), encoding="utf-8")
    return table, impact


def _decision_tuple(result):
    return {
        "amount": str(result.amount_safe_to_pay),
        "status": result.affordability_status.value,
        "method": result.recommended_payment_method.value,
        "plan": format_payment_plan(result.payment_plan),
        "earliest": result.earliest_date_for_full_payment.isoformat() if result.earliest_date_for_full_payment else "",
        "changes": format_spending_changes(result.spending_changes_needed),
    }


def part3_ab(repo, cache, client):
    on = forecast_config_from_profiles(repo.dataset.profiles, strict_unresolved_amounts=True, essential_spend_enabled=True)
    off = forecast_config_from_profiles(repo.dataset.profiles, strict_unresolved_amounts=True, essential_spend_enabled=False)
    rows = []
    changed = []
    for sample in repo.dataset.sample_requests:
        request = sample.request
        _, state, _ = resolve_request_state(repo, request, client=client, cache=cache)
        options = repo.payment_options_for_request(request.request_id)
        cap_on = compute_capacity(state, request, on)
        cap_off = compute_capacity(state, request, off)
        dec_on = decide(state, request, options, on, capacity=cap_on)
        dec_off = decide(state, request, options, off, capacity=cap_off)
        a = _decision_tuple(dec_on)
        b = _decision_tuple(dec_off)
        label = {
            "amount": str(sample.decision.amount_safe_to_pay),
            "status": sample.decision.affordability_status.value,
            "method": sample.decision.recommended_payment_method.value,
            "plan": sample.decision.payment_plan,
            "earliest": sample.decision.earliest_date_for_full_payment.isoformat()
            if sample.decision.earliest_date_for_full_payment else "",
            "changes": sample.decision.spending_changes_needed,
        }
        profiles = build_category_profiles(state, on, cap_on.baseline.all_entries)
        reserve_detail = [
            {
                "category": p.category,
                "residual": str(p.residual_reserve),
                "expected": str(p.projected_budget),
                "explicit": str(p.explicit_projected_amount),
                "n": len(p.historical_event_ids),
                "confidence": p.confidence.value,
                "ids": list(p.historical_event_ids[:8]),
            }
            for p in profiles
            if p.residual_reserve > 0
        ]
        row = {
            "request_id": request.request_id,
            "on": a,
            "off": b,
            "label": label,
            "amount_match_on": a["amount"] == label["amount"],
            "amount_match_off": b["amount"] == label["amount"],
            "status_match_on": a["status"] == label["status"],
            "status_match_off": b["status"] == label["status"],
            "method_match_on": a["method"] == label["method"],
            "method_match_off": b["method"] == label["method"],
            "plan_match_on": a["plan"] == label["plan"],
            "plan_match_off": b["plan"] == label["plan"],
            "earliest_match_on": a["earliest"] == label["earliest"],
            "earliest_match_off": b["earliest"] == label["earliest"],
            "changes_match_on": a["changes"] == label["changes"],
            "changes_match_off": b["changes"] == label["changes"],
            "ab_changed": a != b,
            "reserve_detail": reserve_detail,
            "on_vs_label_amount_err": str(Decimal(a["amount"]) - Decimal(label["amount"])),
            "off_vs_label_amount_err": str(Decimal(b["amount"]) - Decimal(label["amount"])),
        }
        rows.append(row)
        if a != b:
            changed.append(row)
    metrics = {
        "n": len(rows),
        "ab_amount_same": sum(1 for r in rows if r["on"]["amount"] == r["off"]["amount"]),
        "ab_status_same": sum(1 for r in rows if r["on"]["status"] == r["off"]["status"]),
        "ab_method_same": sum(1 for r in rows if r["on"]["method"] == r["off"]["method"]),
        "ab_plan_same": sum(1 for r in rows if r["on"]["plan"] == r["off"]["plan"]),
        "ab_earliest_same": sum(1 for r in rows if r["on"]["earliest"] == r["off"]["earliest"]),
        "ab_changes_same": sum(1 for r in rows if r["on"]["changes"] == r["off"]["changes"]),
        "ab_full_same": sum(1 for r in rows if r["on"] == r["off"]),
        "label_amount_on": sum(1 for r in rows if r["amount_match_on"]),
        "label_amount_off": sum(1 for r in rows if r["amount_match_off"]),
        "label_status_on": sum(1 for r in rows if r["status_match_on"]),
        "label_status_off": sum(1 for r in rows if r["status_match_off"]),
        "label_method_on": sum(1 for r in rows if r["method_match_on"]),
        "label_method_off": sum(1 for r in rows if r["method_match_off"]),
        "label_plan_on": sum(1 for r in rows if r["plan_match_on"]),
        "label_plan_off": sum(1 for r in rows if r["plan_match_off"]),
        "label_earliest_on": sum(1 for r in rows if r["earliest_match_on"]),
        "label_earliest_off": sum(1 for r in rows if r["earliest_match_off"]),
        "label_changes_on": sum(1 for r in rows if r["changes_match_on"]),
        "label_changes_off": sum(1 for r in rows if r["changes_match_off"]),
        "label_full_on": sum(1 for r in rows if all(r[k] for k in (
            "amount_match_on", "status_match_on", "method_match_on",
            "plan_match_on", "earliest_match_on", "changes_match_on",
        ))),
        "label_full_off": sum(1 for r in rows if all(r[k] for k in (
            "amount_match_off", "status_match_off", "method_match_off",
            "plan_match_off", "earliest_match_off", "changes_match_off",
        ))),
        "changed_ids": [r["request_id"] for r in changed],
    }
    (OUT / "part3_ab.json").write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    return metrics, rows, changed


def part4_inventory(repo):
    blanks = [e for e in repo.dataset.events if e.amount is None]
    images = repo.dataset.images
    eval_ids = {r.request_id for r in repo.dataset.evaluation_requests}
    sample_ids = {s.request.request_id for s in repo.dataset.sample_requests}
    rows = []
    for event in blanks:
        imgs = [img for img in images if img.related_event_id == event.event_id]
        reqs = []
        for img in imgs:
            if img.request_id:
                reqs.append(img.request_id)
        rows.append({
            "event_id": event.event_id,
            "user_id": event.user_id,
            "type": event.event_type.value,
            "status": event.status.value,
            "desc": event.description,
            "images": [img.image_id for img in imgs],
            "image_request_ids": reqs,
            "eval_requests": [rid for rid in reqs if rid in eval_ids],
            "sample_requests": [rid for rid in reqs if rid in sample_ids],
        })
    (OUT / "part4_blank_events.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    repo = DatasetRepository(load_dataset())
    config = forecast_config_from_profiles(repo.dataset.profiles, strict_unresolved_amounts=True)
    cache = EvidenceCache()
    client = ModelClient(api_key="unused")
    print("PART1 request_11...")
    p11 = part1_request11(repo, cache, client, config)
    print("  latest dining", p11["combined_dining_cadence"])
    print("  predicted", p11["predicted"])
    print("  label", p11["label"])
    print("PART1 scan...")
    rows, recon, by_state = part1_scan(repo, cache, client, config)
    inconsist = [r for r in rows if r["inconsistency"]]
    prod_ok = [r for r in rows if r["optimizer_change_allowed"]]
    print(f"  preference-legal flexible series-rows: {len(rows)}")
    print(f"  production optimizer allowed: {len(prod_ok)}")
    print(f"  preference-legal but not in forecast: {len(inconsist)}")
    print(f"  category reconstructions (split desc, regular combined): {len(recon)}")
    print("PART2 duplicates...")
    table, impact = part2_duplicates(repo, cache, client, config, by_state)
    print(f"  chains: {len(table)} impact rows: {len(impact)} changed: {sum(1 for i in impact if i['changed'])}")
    print("PART3 A/B...")
    metrics, _rows, changed = part3_ab(repo, cache, client)
    print(json.dumps(metrics, indent=2))
    print("PART4 blanks...")
    blanks = part4_inventory(repo)
    print(f"  blank events: {len(blanks)}")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
