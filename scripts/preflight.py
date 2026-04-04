#!/usr/bin/env python
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from src.utils.resource_guard import select_profile
from src.utils.system import get_system_snapshot


MODULES = [
    "torch",
    "pettingzoo",
    "supersuit",
    "gymnasium",
    "pygame",
    "matplotlib",
    "numpy",
    "pandas",
]


def module_status(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def main() -> None:
    imports = {name: module_status(name) for name in MODULES}
    system = get_system_snapshot(ROOT)
    deps_ok = all(imports[name]["ok"] for name in ["pettingzoo", "supersuit", "gymnasium", "pygame", "numpy", "matplotlib"])
    profile = select_profile(
        free_disk_gb=system["free_disk_gb"],
        deps_ok=deps_ok,
        logical_cores=system["logical_cores"],
        torch_ok=imports["torch"]["ok"],
        mpe_ok=imports["pettingzoo"]["ok"] and imports["gymnasium"]["ok"],
    )
    out = {
        **system,
        "imports": imports,
        "profile": profile,
        "python_executable": sys.executable,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "preflight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [
        "# PREFLIGHT",
        f"- OS: {out['os']}",
        f"- Python: {out['python']} ({out['python_executable']})",
        f"- Apple Silicon: {out['apple_silicon']}",
        f"- Logical cores: {out['logical_cores']}",
        f"- RAM GB: {out['ram_gb']}",
        f"- Free disk GB: {out['free_disk_gb']}",
        f"- Repo size MB: {out['repo_size_mb']}",
        f"- Battery: {out['battery']}",
        f"- Imports: {imports}",
        f"- Selected profile: {profile}",
    ]
    (results_dir / "PREFLIGHT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (results_dir / "ACTIVE_PROFILE.txt").write_text(profile + "\n", encoding="utf-8")
    print(profile)


if __name__ == "__main__":
    main()
