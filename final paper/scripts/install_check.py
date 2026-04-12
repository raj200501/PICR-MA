#!/usr/bin/env python
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ["pytest", "numpy", "pandas", "matplotlib", "torch", "pettingzoo", "supersuit", "gymnasium", "pygame"]


def main() -> None:
    status = {"python_executable": sys.executable}
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            status[module_name] = getattr(module, "__version__", "ok")
        except Exception as exc:
            status[module_name] = f"missing:{type(exc).__name__}"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "install_check.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
