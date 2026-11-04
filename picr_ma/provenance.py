"""Which code and software produced an artifact.

Evaluation, benchmark and analysis outputs record ``provenance()``: the git
commit, whether the code (``picr_ma/``, ``scripts/``, ``configs/``) had
uncommitted changes, and the Python, platform, CPU, NumPy and PyTorch versions.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ("picr_ma", "scripts", "configs")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def provenance() -> dict:
    import numpy
    import torch

    status = _git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "code_modified": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }
