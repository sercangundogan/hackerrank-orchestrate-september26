"""Detect recurring-series CANDIDATES. Do not expand future occurrences."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from statistics import median

from data.models import EventDirection, EventType, FinancialProfile, Flexibility
from finance.income import classify_income_text, is_base_payroll
from finance.models import (
    AmountBehavior,
    Cadence,
    CadenceConfidence,
    NormalizedCashEvent,
    RecurringSeriesCandidate,
)

_RECURRING_FLEXIBILITY = {
    Flexibility.REDUCIBLE,
    Flexibility.STOPPABLE,
    Flexibility.REDUCIBLE_OR_STOPPABLE,
}


def normalize_description(text: str) -> str:
    return " ".join(text.strip().lower().split())


def infer_cadence(dates: tuple[date, ...]) -> tuple[Cadence, int | None, CadenceConfidence, str]:
    ordered = tuple(sorted(dates))
    if len(ordered) < 2:
        return Cadence.UNKNOWN, None, CadenceConfidence.LOW, "fewer than 2 observations"
    gaps = [(b - a).days for a, b in zip(ordered, ordered[1:])]
    median_gap = int(median(gaps))
    spread = max(gaps) - min(gaps)
    consistent = spread <= 3

    if 6 <= median_gap <= 8 and consistent:
        cadence = Cadence.WEEKLY
    elif 13 <= median_gap <= 15 and consistent:
        cadence = Cadence.BIWEEKLY
    elif 28 <= median_gap <= 32 and consistent:
        cadence = Cadence.MONTHLY
    else:
        cadence = Cadence.IRREGULAR

    if cadence is Cadence.IRREGULAR:
        confidence = CadenceConfidence.LOW
        reason = f"median gap {median_gap}d with spread {spread}d is irregular"
    elif len(ordered) >= 4 and consistent:
        confidence = CadenceConfidence.HIGH
        reason = f"{len(ordered)} observations, median gap {median_gap}d, spread {spread}d"
    elif len(ordered) == 3 and consistent:
        confidence = CadenceConfidence.MEDIUM
        reason = f"3 observations, median gap {median_gap}d, spread {spread}d"
    else:
        confidence = CadenceConfidence.LOW
        reason = f"{len(ordered)} observations near {cadence.value} spacing"
    return cadence, median_gap, confidence, reason


def _amount_behavior(
    amounts: tuple[Decimal | None, ...],
) -> tuple[AmountBehavior, Decimal | None]:
    known = tuple(amount for amount in amounts if amount is not None)
    if not known:
        return AmountBehavior.UNKNOWN, None
    if all(amount == known[0] for amount in known):
        return AmountBehavior.FIXED, known[-1]
    return AmountBehavior.VARIABLE, None


def _eligible_history(event: NormalizedCashEvent) -> bool:
    if event.status.value != "settled":
        return False
    if not event.is_historical:
        return False
    if event.event_type in {
        EventType.REFUND,
        EventType.INVESTMENT_SALE,
        EventType.INVESTMENT_VALUATION,
    }:
        return False
    if event.ignore_reason is not None and event.ignore_reason.value in {
        "cancelled",
        "failed",
        "unrealized",
    }:
        return False
    return True


def _likely_recurring(
    *,
    event_type: EventType,
    category: str,
    flexibility: Flexibility,
    cadence: Cadence,
    confidence: CadenceConfidence,
    observation_count: int,
) -> bool:
    if confidence in {CadenceConfidence.HIGH, CadenceConfidence.MEDIUM}:
        return True
    if event_type is EventType.SUBSCRIPTION:
        return True
    if category == "salary" and observation_count >= 1:
        return True
    if flexibility in _RECURRING_FLEXIBILITY:
        # Sparse flexible series remain identifiable for later spending changes
        # (sample request_11 reduces a 2-observation dining series).
        return True
    if cadence in {Cadence.WEEKLY, Cadence.BIWEEKLY, Cadence.MONTHLY} and observation_count >= 2:
        return True
    return False


def build_recurring_series_candidates(
    *,
    profile: FinancialProfile,
    historical_events: tuple[NormalizedCashEvent, ...],
    confirmed_scheduled_income: tuple[NormalizedCashEvent, ...],
    requires_message_resolution: bool,
) -> tuple[RecurringSeriesCandidate, ...]:
    grouped: dict[tuple[str, str, EventDirection], list[NormalizedCashEvent]] = defaultdict(list)
    for event in historical_events:
        if not _eligible_history(event):
            continue
        key = (event.category, normalize_description(event.description), event.direction)
        grouped[key].append(event)

    scheduled_by_key: dict[tuple[str, str, EventDirection], list[NormalizedCashEvent]] = defaultdict(
        list
    )
    for event in confirmed_scheduled_income:
        key = (event.category, normalize_description(event.description), event.direction)
        scheduled_by_key[key].append(event)
        # Also attach confirmed salary to the historical salary series even when
        # the scheduled description differs ("Next confirmed salary" vs "Payroll credit").
        if event.category == "salary":
            event_subtype = classify_income_text(event.description, event.category)
            matches = [
                hist_key
                for hist_key in grouped
                if hist_key[0] == "salary" and hist_key[2] is EventDirection.CREDIT
                and is_base_payroll(classify_income_text(hist_key[1], hist_key[0]))
                and (
                    is_base_payroll(event_subtype)
                    or "confirmed salary" in (event.description or "").lower()
                )
            ]
            # Attach a generic next-salary row to the single best base series only.
            if len(matches) > 1 and "confirmed salary" in (event.description or "").lower():
                matches = sorted(
                    matches,
                    key=lambda key: (
                        0 if "payroll" in key[1] or "base" in key[1] or "primary" in key[1] else 1,
                        -len(grouped[key]),
                        key[1],
                    ),
                )[:1]
            for hist_key in matches:
                scheduled_by_key[hist_key].append(event)

    candidates: list[RecurringSeriesCandidate] = []
    for (category, normalized, direction), members in grouped.items():
        members = sorted(members, key=lambda item: (item.cash_date or item.event_date, item.source_event_id))
        dates = tuple((item.cash_date or item.event_date) for item in members)
        amounts = tuple(item.amount_home_currency for item in members)
        cadence, cadence_days, confidence, cadence_reason = infer_cadence(dates)
        amount_behavior, representative = _amount_behavior(amounts)
        latest = members[-1]
        scheduled = tuple(
            item.source_event_id
            for item in scheduled_by_key.get((category, normalized, direction), ())
        )
        # Generic "next confirmed salary" is attached above to the single best
        # base-payroll series only. Do not copy it onto every salary stream.

        flexibility = latest.flexibility
        likely = _likely_recurring(
            event_type=latest.event_type,
            category=category,
            flexibility=flexibility,
            cadence=cadence,
            confidence=confidence,
            observation_count=len(members),
        )
        reason_parts = [cadence_reason]
        if latest.event_type is EventType.SUBSCRIPTION:
            reason_parts.append("subscription type")
        if flexibility in _RECURRING_FLEXIBILITY:
            reason_parts.append(f"flexibility={flexibility.value}")
        if scheduled:
            reason_parts.append("has scheduled confirmed income")

        candidates.append(
            RecurringSeriesCandidate(
                user_id=profile.user_id,
                category=category,
                normalized_description=normalized,
                original_description=latest.description,
                event_ids=tuple(item.source_event_id for item in members),
                direction=direction,
                event_type=latest.event_type,
                observed_dates=dates,
                observed_amounts_home_currency=amounts,
                inferred_cadence=cadence,
                inferred_cadence_days=cadence_days,
                cadence_confidence=confidence,
                amount_behavior=amount_behavior,
                representative_amount=representative,
                flexibility=flexibility,
                minimum_allowed_amount=latest.minimum_allowed_amount,
                is_protected_category=category in profile.expense_categories_to_protect,
                profile_allows_reduce=category
                in profile.expense_categories_user_is_willing_to_reduce,
                profile_allows_stop=category
                in profile.expense_categories_user_is_willing_to_stop,
                likely_recurring=likely,
                scheduled_confirmed_event_ids=scheduled,
                requires_message_resolution=requires_message_resolution
                and category == "salary",
                reason="; ".join(reason_parts),
            )
        )

    # Salary with only a scheduled confirmed row and no historical group still
    # gets metadata so Phase 3 can see the confirmed payday without inventing more.
    if not any(candidate.category == "salary" for candidate in candidates):
        salary_scheduled = [
            event for event in confirmed_scheduled_income if event.category == "salary"
        ]
        if salary_scheduled:
            latest = salary_scheduled[-1]
            candidates.append(
                RecurringSeriesCandidate(
                    user_id=profile.user_id,
                    category="salary",
                    normalized_description=normalize_description(latest.description),
                    original_description=latest.description,
                    event_ids=(),
                    direction=EventDirection.CREDIT,
                    event_type=EventType.INCOME,
                    observed_dates=(),
                    observed_amounts_home_currency=(),
                    inferred_cadence=Cadence.UNKNOWN,
                    inferred_cadence_days=None,
                    cadence_confidence=CadenceConfidence.LOW,
                    amount_behavior=AmountBehavior.UNKNOWN,
                    representative_amount=None,
                    flexibility=latest.flexibility,
                    minimum_allowed_amount=None,
                    is_protected_category=False,
                    profile_allows_reduce=False,
                    profile_allows_stop=False,
                    likely_recurring=True,
                    scheduled_confirmed_event_ids=tuple(
                        item.source_event_id for item in salary_scheduled
                    ),
                    requires_message_resolution=requires_message_resolution,
                    reason="confirmed scheduled salary without historical series",
                )
            )

    candidates.sort(
        key=lambda item: (
            item.category,
            item.normalized_description,
            item.direction.value,
        )
    )
    return tuple(candidates)
