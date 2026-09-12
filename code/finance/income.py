"""Future-income classification and confirmation policy.

Expenses may be inferred conservatively from history. Income may increase
future cash only when the problem rules and evidence support it.
"""

from __future__ import annotations

from enum import Enum

from finance.adjustments import ForecastAdjustment, ForecastAdjustmentKind
from finance.forecast_models import SalaryProjectionMode
from finance.models import Cadence, CadenceConfidence, RecurringSeriesCandidate


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


class IncomeSubtype(str, Enum):
    BASE_SALARY = "base_salary"
    COMMISSION = "commission"
    BONUS = "bonus"
    REFUND = "refund"
    REIMBURSEMENT = "reimbursement"
    GIG_PAYOUT = "gig_payout"
    WINDFALL = "windfall"
    SALE_PROCEEDS = "sale_proceeds"
    FINAL_PAYROLL = "final_payroll"
    ONE_TIME_ADJUSTMENT = "one_time_adjustment"
    OTHER = "other"


class IncomeConfidence(str, Enum):
    CONFIRMED = "confirmed"
    CONTINUATION_SUPPORTED = "continuation_supported"
    UNCONFIRMED = "unconfirmed"
    STOPPED = "stopped"


_NEVER_PROJECT = {
    IncomeSubtype.COMMISSION,
    IncomeSubtype.BONUS,
    IncomeSubtype.REFUND,
    IncomeSubtype.REIMBURSEMENT,
    IncomeSubtype.GIG_PAYOUT,
    IncomeSubtype.WINDFALL,
    IncomeSubtype.SALE_PROCEEDS,
    IncomeSubtype.ONE_TIME_ADJUSTMENT,
}

_CONFIRMING_KINDS = {
    ForecastAdjustmentKind.SALARY_AMOUNT,
    ForecastAdjustmentKind.SALARY_TEMPORARY,
    ForecastAdjustmentKind.SALARY_PAYDAY,
    ForecastAdjustmentKind.START_SALARY,
    ForecastAdjustmentKind.CONFIRM_INCOME,
}


def classify_income_text(description: str, category: str | None = None) -> IncomeSubtype:
    """General subtype from description/category. No request IDs."""
    text = f"{category or ''} {description or ''}".lower()
    if any(token in text for token in ("commission", "incentive")):
        return IncomeSubtype.COMMISSION
    if "bonus" in text:
        return IncomeSubtype.BONUS
    if "refund" in text:
        return IncomeSubtype.REFUND
    if "reimburse" in text:
        return IncomeSubtype.REIMBURSEMENT
    if any(token in text for token in ("windfall", "prize", "lottery")):
        return IncomeSubtype.WINDFALL
    if category == "investment" or "unrealized" in text:
        return IncomeSubtype.SALE_PROCEEDS
    if any(token in text for token in ("gig", "invoice payout", "withdrawable")):
        return IncomeSubtype.GIG_PAYOUT
    if "final" in text and any(token in text for token in ("payroll", "salary", "employer")):
        return IncomeSubtype.FINAL_PAYROLL
    if any(token in text for token in ("arrear", "back pay", "one-time", "one time")):
        return IncomeSubtype.ONE_TIME_ADJUSTMENT
    if category == "salary" or any(
        token in text
        for token in (
            "payroll",
            "salary",
            "wage",
            "household income",
            "base pay",
            "base salary",
        )
    ):
        return IncomeSubtype.BASE_SALARY
    return IncomeSubtype.OTHER


def classify_series(series: RecurringSeriesCandidate) -> IncomeSubtype:
    return classify_income_text(series.original_description, series.category)


def is_base_payroll(subtype: IncomeSubtype) -> bool:
    return subtype is IncomeSubtype.BASE_SALARY


def is_salary_like_credit(series: RecurringSeriesCandidate) -> bool:
    return series.category == "salary" and series.direction.value == "credit"


def never_project_subtype(subtype: IncomeSubtype) -> bool:
    return subtype in _NEVER_PROJECT or subtype is IncomeSubtype.FINAL_PAYROLL


def confirming_adjustments(
    adjustments: tuple[ForecastAdjustment, ...],
) -> tuple[ForecastAdjustment, ...]:
    return tuple(item for item in adjustments if item.kind in _CONFIRMING_KINDS)


def employment_stopped(adjustments: tuple[ForecastAdjustment, ...]) -> bool:
    return any(item.kind is ForecastAdjustmentKind.STOP_SALARY_PROJECTION for item in adjustments)


def latest_salary_is_final(
    candidates: tuple[RecurringSeriesCandidate, ...],
) -> bool:
    """A later final-payroll event stops every salary stream, not just itself."""
    latest_date = None
    latest_subtype = None
    for candidate in candidates:
        if candidate.category != "salary" or not candidate.observed_dates:
            continue
        last = max(candidate.observed_dates)
        subtype = classify_series(candidate)
        if latest_date is None or last > latest_date:
            latest_date = last
            latest_subtype = subtype
        elif last == latest_date and subtype is IncomeSubtype.FINAL_PAYROLL:
            latest_subtype = subtype
    return latest_subtype is IncomeSubtype.FINAL_PAYROLL


def _notes_target_series(item: ForecastAdjustment, series: RecurringSeriesCandidate) -> bool:
    blob = " ".join((item.notes, item.target_description or "", item.category or "")).lower()
    if not blob.strip():
        return False
    desc = series.original_description.lower()
    norm = series.normalized_description
    if item.target_description:
        return _norm(item.target_description) == norm
    if any(token in blob for token in ("commission", "bonus")) and classify_series(series) in {
        IncomeSubtype.COMMISSION,
        IncomeSubtype.BONUS,
    }:
        return True
    if any(token in blob for token in ("second", "other household")) and "second" in desc:
        return True
    if any(token in blob for token in ("primary", "base", "payroll")) and is_base_payroll(
        classify_series(series)
    ):
        return True
    return False


def amendment_targets_series(
    item: ForecastAdjustment,
    series: RecurringSeriesCandidate,
    *,
    candidates: tuple[RecurringSeriesCandidate, ...],
) -> bool:
    """Whether a salary amendment may change this series.

    Safer default: only the relevant base-payroll stream. Combined
    compensation is applied only when evidence says so.
    """
    if item.kind not in {
        ForecastAdjustmentKind.SALARY_AMOUNT,
        ForecastAdjustmentKind.SALARY_TEMPORARY,
        ForecastAdjustmentKind.SALARY_PAYDAY,
        ForecastAdjustmentKind.START_SALARY,
    }:
        return False
    subtype = classify_series(series)
    if never_project_subtype(subtype) and "combined" not in (item.notes or "").lower():
        return False
    if item.income_subtype and item.income_subtype not in {subtype.value, "base_salary"}:
        return False
    if item.income_subtype == "base_salary" and not is_base_payroll(subtype):
        return False
    if item.target_description:
        return _norm(item.target_description) == series.normalized_description
    if _notes_target_series(item, series):
        return True
    base_series = tuple(cand for cand in candidates if is_base_payroll(classify_series(cand)))
    if not is_base_payroll(subtype):
        return "combined" in (item.notes or "").lower()
    if len(base_series) <= 1:
        return True
    # Ambiguous multi-stream household: prefer the scheduled/primary payroll only.
    scored = sorted(
        base_series,
        key=lambda cand: (
            0 if cand.scheduled_confirmed_event_ids else 1,
            0 if "primary" in cand.original_description.lower() or "base" in cand.original_description.lower() else 1,
            0 if "payroll" in cand.original_description.lower() else 1,
            -len(cand.event_ids),
            cand.normalized_description,
        ),
    )
    return scored[0].normalized_description == series.normalized_description


def series_has_explicit_schedule(series: RecurringSeriesCandidate) -> bool:
    return bool(series.scheduled_confirmed_event_ids)


def user_has_explicit_salary_confirmation(
    *,
    series: RecurringSeriesCandidate,
    adjustments: tuple[ForecastAdjustment, ...],
    candidates: tuple[RecurringSeriesCandidate, ...],
) -> bool:
    if series.scheduled_confirmed_event_ids:
        return True
    for item in confirming_adjustments(adjustments):
        if item.kind is ForecastAdjustmentKind.CONFIRM_INCOME and item.category not in {
            None,
            "salary",
        }:
            continue
        if item.kind in {
            ForecastAdjustmentKind.SALARY_AMOUNT,
            ForecastAdjustmentKind.SALARY_TEMPORARY,
            ForecastAdjustmentKind.SALARY_PAYDAY,
            ForecastAdjustmentKind.START_SALARY,
        } and not amendment_targets_series(item, series, candidates=candidates):
            continue
        return True
    return False


def strong_base_history(series: RecurringSeriesCandidate) -> bool:
    if not is_base_payroll(classify_series(series)):
        return False
    if series.inferred_cadence is not Cadence.MONTHLY:
        return False
    return series.cadence_confidence is CadenceConfidence.HIGH


def salary_generation_allowed(
    series: RecurringSeriesCandidate,
    *,
    mode: SalaryProjectionMode,
    adjustments: tuple[ForecastAdjustment, ...],
    candidates: tuple[RecurringSeriesCandidate, ...],
) -> tuple[bool, IncomeConfidence, str]:
    """Decide whether a salary-like series may generate future credits."""
    subtype = classify_series(series)
    if employment_stopped(adjustments) or latest_salary_is_final(candidates):
        return False, IncomeConfidence.STOPPED, "employment_or_salary_stop"
    if never_project_subtype(subtype):
        return False, IncomeConfidence.UNCONFIRMED, f"unconfirmed_{subtype.value}"
    if not is_salary_like_credit(series) and not is_base_payroll(subtype):
        return False, IncomeConfidence.UNCONFIRMED, "non_salary_credit"

    confirmed = user_has_explicit_salary_confirmation(
        series=series, adjustments=adjustments, candidates=candidates
    )
    scheduled = series_has_explicit_schedule(series)
    strong = strong_base_history(series)

    if mode is SalaryProjectionMode.STRICT_CONFIRMED:
        if scheduled or confirmed:
            return True, IncomeConfidence.CONFIRMED, "strict_confirmed_only"
        return False, IncomeConfidence.UNCONFIRMED, "no_explicit_confirmation"

    if mode is SalaryProjectionMode.CONFIRMED_THEN_CONTINUE:
        if scheduled or confirmed:
            if strong or scheduled or confirmed:
                return True, IncomeConfidence.CONTINUATION_SUPPORTED, "confirmed_then_continue"
        return False, IncomeConfidence.UNCONFIRMED, "history_only_not_confirmed"

    # STRONG_BASE_SALARY and legacy scheduled-plus-regular-history: HIGH base only.
    if strong:
        confidence = (
            IncomeConfidence.CONFIRMED
            if scheduled or confirmed
            else IncomeConfidence.CONTINUATION_SUPPORTED
        )
        return True, confidence, "strong_base_salary_history"
    if scheduled or confirmed:
        return True, IncomeConfidence.CONTINUATION_SUPPORTED, "confirmed_weak_history"
    return False, IncomeConfidence.UNCONFIRMED, "weak_unconfirmed_history"


def continue_start_salary(mode: SalaryProjectionMode) -> bool:
    return mode is not SalaryProjectionMode.STRICT_CONFIRMED
