"""Load the dataset and optionally generate root output.csv."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.loader import load_dataset
from data.repository import DatasetRepository
from data.validation import DatasetValidationError, validate_dataset
from ai.client import ModelClient
from evidence.cache import EvidenceCache
from evidence.pipeline import extract_evidence, resolve_request_state
from config import OUTPUT_CSV, REPO_ROOT
from decision.capacity import compute_capacity
from decision.generate import (
    GeneratedDecision,
    diagnostics,
    generate_evaluation_decisions,
    render_output_csv,
    validate_generated,
    write_output_csv,
)
from decision.output_validator import OutputValidationError
from usage.report import USAGE_REPORT_PATH, render_usage_report, write_usage_report
from finance.diagnostics import (
    format_capacity_diagnostics,
    format_evidence_diagnostics,
    format_financial_state_diagnostics,
    format_forecast_diagnostics,
)
from finance.essential_spending import forecast_config_from_profiles
from finance.forecast import forecast_financial_state
from finance.normalization import finance_request_by_id, normalize_financial_state
from usage.tracker import UsageTracker


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buy or Wait? dataset smoke runner")
    parser.add_argument(
        "--debug-request",
        metavar="REQUEST_ID",
        help="Print Phase 2 normalization diagnostics for one request. Does not write output.csv.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Print Phase 3 90-day forecast diagnostics. Requires --debug-request.",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Extract and apply message/image evidence. Requires --debug-request unless --extract-evidence.",
    )
    parser.add_argument(
        "--extract-evidence",
        action="store_true",
        help="Extract evidence for every evaluation and sample request. Does not write output.csv.",
    )
    parser.add_argument(
        "--capacity",
        action="store_true",
        help="Print Phase 5A capacity diagnostics. Requires --debug-request. Does not write output.csv.",
    )
    parser.add_argument(
        "--generate-output",
        action="store_true",
        help="Generate repository-root output.csv for all evaluation requests. Does not modify dataset/output.csv.",
    )
    return parser.parse_args(argv)


def _request_usage_stats(
    tracker: UsageTracker, request_ids: list[str]
) -> tuple[int, int, int, int]:
    paid: set[str] = set()
    cached_images: set[str] = set()
    paid_messages: set[str] = set()
    for item in tracker.records:
        rid = item.request_id
        if not rid:
            continue
        if item.success and not item.cache_hit:
            paid.add(rid)
            if "message" in item.purpose:
                paid_messages.add(rid)
        if item.cache_hit and "image" in item.purpose:
            cached_images.add(rid)
    id_set = set(request_ids)
    zero_paid = len(id_set - paid)
    deterministic_messages = len(id_set - paid_messages)
    return zero_paid, len(cached_images & id_set), deterministic_messages, len(paid & id_set)


def _suspicious(stats: dict[str, object]) -> list[str]:
    n = int(stats["n"])
    method = stats["method"]
    flags: list[str] = []
    not_affordable = int(stats["status"].get("not_affordable", 0))
    if n and not_affordable / n >= 0.95:
        flags.append(">=95% not_affordable")
    if method.get("installments", 0) == 0:
        flags.append("zero installments")
    if int(stats["zero_safe_amounts"]) == n:
        flags.append("all safe amounts are zero")
    return flags


def _print_sanity(decisions: tuple[GeneratedDecision, ...]) -> None:
    wanted = (
        ("affordable_now", "full_payment"),
        ("affordable_with_plan", "installments"),
        ("affordable_with_plan", "partial_payment"),
        ("affordable_later", "wait"),
        ("not_affordable", "not_recommended"),
    )
    print()
    print("Sanity sample")
    seen: set[str] = set()
    for status, method in wanted:
        match = next(
            (
                item
                for item in decisions
                if item.result.affordability_status.value == status
                and item.result.recommended_payment_method.value == method
            ),
            None,
        )
        label = f"{status}/{method}"
        if match is None:
            print(f"- {label}: none")
            continue
        seen.add(match.request.request_id)
        row = __import__("decision.output", fromlist=["decision_to_row"]).decision_to_row(
            match.result
        )
        print(
            f"- {label}: {row['request_id']} amount={row['amount_safe_to_pay']} "
            f"plan={row['payment_plan']} changes={row['spending_changes_needed']}"
        )
    changed = next((item for item in decisions if item.result.spending_changes_needed), None)
    if changed is not None and changed.request.request_id not in seen:
        row = __import__("decision.output", fromlist=["decision_to_row"]).decision_to_row(
            changed.result
        )
        print(
            f"- spending_changes: {row['request_id']} {row['spending_changes_needed']}"
        )
    imaged = next((item for item in decisions if item.bundle.image_extractions), None)
    if imaged is not None:
        print(
            f"- image evidence: {imaged.request.request_id} "
            f"images={len(imaged.bundle.image_extractions)} failures={len(imaged.bundle.failures)}"
        )


def _generate_output(repository, dataset, forecast_config) -> int:
    run_id = "final-output-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = REPO_ROOT / "evaluation" / "final_run_usage.jsonl"
    tracker = UsageTracker(persist_path=ledger)
    cache = EvidenceCache()
    client = ModelClient(tracker=tracker)
    print()
    print(f"Final run id: {run_id}")
    print("Generating evaluation decisions (cache preserved)...")
    first = generate_evaluation_decisions(
        repository, cache=cache, client=client, tracker=tracker, config=forecast_config
    )
    first_csv = render_output_csv(first)
    repro_tracker = UsageTracker()
    second = generate_evaluation_decisions(
        repository,
        cache=cache,
        client=ModelClient(tracker=repro_tracker),
        tracker=repro_tracker,
        config=forecast_config,
    )
    if first_csv != render_output_csv(second):
        print("Reproducibility check FAILED: two identical runs produced different CSV.", file=sys.stderr)
        return 3
    print("Reproducibility check: PASS")
    stats = diagnostics(first)
    print()
    print("Distributions")
    print(f"  status: {stats['status']}")
    print(f"  method: {stats['method']}")
    print(f"  currency: {stats['currency']}")
    print(f"  spending-change rows: {stats['spending_change_rows']}")
    print(f"  unresolved evidence notes: {stats['unresolved_evidence']}")
    print(f"  zero safe amounts: {stats['zero_safe_amounts']}")
    print(f"  method by request type: {stats['method_by_type']}")
    flags = _suspicious(stats)
    if flags:
        print("Suspicious distribution: " + "; ".join(flags), file=sys.stderr)
        return 5
    try:
        write_output_csv(OUTPUT_CSV, first)
        validate_generated(OUTPUT_CSV, repository, first, forecast_config)
    except OutputValidationError as exc:
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
        print(f"output validation failed: {exc}", file=sys.stderr)
        return 4
    tracker.persist_overwrite()
    request_ids = [item.request_id for item in dataset.evaluation_requests]
    zero_paid, cached_images, det_messages, new_calls = _request_usage_stats(
        tracker, request_ids
    )
    report = render_usage_report(
        tracker,
        evaluation_requests=len(request_ids),
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        zero_paid_requests=zero_paid,
        cached_image_requests=cached_images,
        deterministic_message_requests=det_messages,
        new_model_call_requests=new_calls,
    )
    write_usage_report(USAGE_REPORT_PATH, report)
    print()
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {USAGE_REPORT_PATH}")
    print(f"Final-run usage: {tracker.summary()}")
    _print_sanity(first)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        dataset = load_dataset()
        validate_dataset(dataset)
        repository = DatasetRepository(dataset)
    except DatasetValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface load/parse failures to the CLI
        print(f"dataset load failed: {exc}", file=sys.stderr)
        return 1

    print("Dataset loaded successfully.")
    print()
    print(f"Profiles: {len(dataset.profiles)}")
    print(f"Evaluation requests: {len(dataset.evaluation_requests)}")
    print(f"Sample requests: {len(dataset.sample_requests)}")
    print(f"Financial events: {len(dataset.events)}")
    print(f"Payment options: {len(dataset.payment_options)}")
    print(f"Messages: {len(dataset.messages)}")
    print(f"Images: {len(dataset.images)}")
    print(f"Exchange rates: {len(dataset.exchange_rates)}")
    print()
    print("Structural validation: PASS")

    if args.forecast and not args.debug_request:
        print("--forecast requires --debug-request REQUEST_ID", file=sys.stderr)
        return 2
    if args.evidence and not args.debug_request and not args.extract_evidence:
        print("--evidence requires --debug-request REQUEST_ID", file=sys.stderr)
        return 2
    if args.capacity and not args.debug_request:
        print("--capacity requires --debug-request REQUEST_ID", file=sys.stderr)
        return 2

    if args.generate_output:
        return _generate_output(repository, dataset, forecast_config_from_profiles(
            dataset.profiles, strict_unresolved_amounts=True
        ))

    tracker = UsageTracker()
    cache = EvidenceCache()
    client = ModelClient(tracker=tracker)
    forecast_config = forecast_config_from_profiles(
        dataset.profiles, strict_unresolved_amounts=True
    )

    if args.extract_evidence:
        request_ids = [item.request_id for item in dataset.evaluation_requests]
        request_ids.extend(item.request.request_id for item in dataset.sample_requests)
        seen: set[str] = set()
        for request_id in request_ids:
            if request_id in seen:
                continue
            seen.add(request_id)
            request = finance_request_by_id(repository, request_id)
            bundle = extract_evidence(
                repository, request, client=client, cache=cache, tracker=tracker
            )
            print(format_evidence_diagnostics(bundle))
            print()
        print("Usage:", tracker.summary())
        return 0

    if args.debug_request:
        try:
            request = finance_request_by_id(repository, args.debug_request)
            if args.evidence or args.capacity:
                base, state, bundle = resolve_request_state(
                    repository, request, client=client, cache=cache, tracker=tracker
                )
            else:
                base = normalize_financial_state(repository, request)
                state = base
                bundle = None
        except Exception as exc:  # noqa: BLE001
            print(f"normalization failed: {exc}", file=sys.stderr)
            return 1
        print()
        print(format_financial_state_diagnostics(base if args.evidence else state))
        if bundle is not None:
            print()
            print(format_evidence_diagnostics(bundle))
        if args.forecast:
            try:
                result = forecast_financial_state(state, forecast_config)
            except Exception as exc:  # noqa: BLE001
                print(f"forecast failed: {exc}", file=sys.stderr)
                return 1
            print()
            print(format_forecast_diagnostics(result))
        if args.capacity:
            try:
                capacity = compute_capacity(state, request, forecast_config)
            except Exception as exc:  # noqa: BLE001
                print(f"capacity failed: {exc}", file=sys.stderr)
                return 1
            print()
            print(format_capacity_diagnostics(capacity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
