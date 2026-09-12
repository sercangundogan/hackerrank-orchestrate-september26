"""Smoke entry point: load, validate, and optionally debug-normalize one request."""

from __future__ import annotations

import argparse
import sys
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
from decision.capacity import compute_capacity
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
    return parser.parse_args(argv)


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
