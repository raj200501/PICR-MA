from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from .io_utils import ensure_dir


def to_csv_gz(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("")
        return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_npz(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        data = np.load(handle, allow_pickle=True)
        return {key: data[key] for key in data.files}
