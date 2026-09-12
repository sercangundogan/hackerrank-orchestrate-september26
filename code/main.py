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
from finance.diagnostics import format_financial_state_diagnostics
from finance.normalization import finance_request_by_id, normalize_financial_state


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buy or Wait? dataset smoke runner")
    parser.add_argument(
        "--debug-request",
        metavar="REQUEST_ID",
        help="Print Phase 2 normalization diagnostics for one request. Does not write output.csv.",
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

    if args.debug_request:
        try:
            request = finance_request_by_id(repository, args.debug_request)
            state = normalize_financial_state(repository, request)
        except Exception as exc:  # noqa: BLE001
            print(f"normalization failed: {exc}", file=sys.stderr)
            return 1
        print()
        print(format_financial_state_diagnostics(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
