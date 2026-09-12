from __future__ import annotations

from pathlib import Path

from data.loader import load_dataset
from data.repository import DatasetRepository
from tests.decision_evaluation import evaluate_sample_decisions, summarize


def test_sample_decision_evaluation_runs() -> None:
    repository = DatasetRepository(load_dataset())
    rows = evaluate_sample_decisions(repository)
    assert len(rows) == 25
    metrics = summarize(rows)
    assert metrics["n"] == 25
    assert metrics["verifier_failures"] == 0


def test_decision_evaluation_is_quarantined() -> None:
    for path in Path("code/decision").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "decision_evaluation" not in text
        assert "sample_requests" not in text
