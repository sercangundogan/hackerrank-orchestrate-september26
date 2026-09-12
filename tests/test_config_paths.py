from __future__ import annotations

from pathlib import Path

from config import CODE_DIR, DATASET_DIR, REPO_ROOT, _resolve_workspace_root


def test_git_layout_resolves_dataset() -> None:
    assert (DATASET_DIR / "requests.csv").is_file()
    assert REPO_ROOT == CODE_DIR.parent
    assert _resolve_workspace_root(CODE_DIR) == REPO_ROOT


def test_flattened_extract_resolves_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "requests.csv").write_text("request_id\n", encoding="utf-8")
    assert _resolve_workspace_root(tmp_path) == tmp_path
