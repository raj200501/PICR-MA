from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.utils.io_utils import read_json


def list_run_dirs(results_dir: str | Path = "results") -> list[Path]:
    return sorted(Path(results_dir).glob("run_*"))


def load_run_summaries(
    results_dir: str | Path = "results",
    env: str | None = None,
    run_tag: str | None = None,
    methods: Iterable[str] | None = None,
) -> list[dict]:
    allowed_methods = set(methods) if methods is not None else None
    rows = []
    for run_dir in list_run_dirs(results_dir):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if env is not None and summary.get("env") != env:
                continue
            if run_tag is not None and summary.get("run_tag") != run_tag:
                continue
            if allowed_methods is not None and summary.get("method") not in allowed_methods:
                continue
            summary["run_dir"] = str(run_dir)
            rows.append(summary)
    return rows


def partner_pool(
    results_dir: str | Path,
    env: str,
    method: str | None = None,
    exclude_run_id: str | None = None,
    run_tag: str | None = None,
) -> list[dict]:
    rows = []
    for summary in load_run_summaries(results_dir, env=env, run_tag=run_tag):
        if summary.get("env") != env:
            continue
        if method is not None and summary.get("method") != method:
            continue
        if exclude_run_id is not None and summary.get("run_id") == exclude_run_id:
            continue
        rows.append(summary)
    return rows
