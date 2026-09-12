"""Repository and dataset path configuration.

Paths are resolved from this file's location so they do not depend on the
process working directory. This module holds constants only — no financial
decision rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "media" / "images"
OUTPUT_CSV = REPO_ROOT / "output.csv"

FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
EVALUATION_REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
OUTPUT_TEMPLATE_CSV = DATASET_DIR / "output.csv"

FORECAST_HORIZON_DAYS = 90
EXPECTED_BLANK_EVENT_AMOUNT_COUNT = 16

OUTPUT_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem locations for every participant-facing dataset file."""

    dataset_dir: Path
    financial_profiles: Path
    evaluation_requests: Path
    sample_requests: Path
    financial_events: Path
    exchange_rates: Path
    payment_options: Path
    messages: Path
    images: Path
    output_template: Path
    images_dir: Path

    @classmethod
    def default(cls) -> DatasetPaths:
        return cls.from_dataset_dir(DATASET_DIR)

    @classmethod
    def from_dataset_dir(cls, dataset_dir: Path) -> DatasetPaths:
        dataset_dir = dataset_dir.resolve()
        return cls(
            dataset_dir=dataset_dir,
            financial_profiles=dataset_dir / "financial_profiles.csv",
            evaluation_requests=dataset_dir / "requests.csv",
            sample_requests=dataset_dir / "sample_requests.csv",
            financial_events=dataset_dir / "financial_events.csv",
            exchange_rates=dataset_dir / "exchange_rates.csv",
            payment_options=dataset_dir / "request_payment_options.csv",
            messages=dataset_dir / "messages.csv",
            images=dataset_dir / "images.csv",
            output_template=dataset_dir / "output.csv",
            images_dir=dataset_dir / "media" / "images",
        )
