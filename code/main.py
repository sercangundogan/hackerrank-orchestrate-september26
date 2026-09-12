"""Phase 1 smoke entry point: load, validate, and summarize the dataset."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.loader import load_dataset
from data.repository import DatasetRepository
from data.validation import DatasetValidationError, validate_dataset


def main() -> int:
    try:
        dataset = load_dataset()
        validate_dataset(dataset)
        DatasetRepository(dataset)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
