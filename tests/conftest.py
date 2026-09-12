from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data.loader import load_dataset
from data.models import Dataset
from data.repository import DatasetRepository


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return load_dataset()


@pytest.fixture(scope="session")
def repository(dataset: Dataset) -> DatasetRepository:
    return DatasetRepository(dataset)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
