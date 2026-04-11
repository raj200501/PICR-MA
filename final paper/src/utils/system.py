import os
import platform
import shutil
import subprocess
from pathlib import Path


def _read_stdout(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def detect_ram_gb() -> float | None:
    if platform.system() == "Darwin":
        out = _read_stdout(["sysctl", "-n", "hw.memsize"])
        if out.isdigit():
            return round(int(out) / 1024**3, 3)
    return None


def detect_battery_status() -> str:
    if platform.system() != "Darwin":
        return "unknown"
    out = _read_stdout(["pmset", "-g", "batt"])
    if "'Battery Power'" in out:
        return "battery"
    if "'AC Power'" in out:
        return "ac"
    return "unknown"


def get_system_snapshot(root: str | Path = ".") -> dict:
    root_path = Path(root)
    total, used, free = shutil.disk_usage(root_path)
    repo_size_mb = round(
        sum(path.stat().st_size for path in root_path.rglob("*") if path.is_file())
        / 1024**2,
        3,
    )
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "apple_silicon": platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"},
        "logical_cores": os.cpu_count() or 1,
        "ram_gb": detect_ram_gb(),
        "free_disk_gb": round(free / 1024**3, 3),
        "repo_size_mb": repo_size_mb,
        "battery": detect_battery_status(),
        "cwd": str(root_path.resolve()),
    }
