from __future__ import annotations

from pathlib import Path

from src.utils.io_utils import read_json


def list_run_dirs(results_dir: str | Path = "results") -> list[Path]:
    return sorted(Path(results_dir).glob("run_*"))


def load_run_summaries(results_dir: str | Path = "results") -> list[dict]:
    rows = []
    for run_dir in list_run_dirs(results_dir):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            summary["run_dir"] = str(run_dir)
            rows.append(summary)
    return rows


def partner_pool(
    results_dir: str | Path,
    env: str,
    method: str,
    exclude_run_id: str | None = None,
) -> list[dict]:
    rows = []
    for summary in load_run_summaries(results_dir):
        if summary.get("env") != env or summary.get("method") != method:
            continue
        if exclude_run_id is not None and summary.get("run_id") == exclude_run_id:
            continue
        rows.append(summary)
    return rows
