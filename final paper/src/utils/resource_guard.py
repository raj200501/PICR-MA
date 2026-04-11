from __future__ import annotations

from .budget import PROFILE_BUDGET_MB


def select_profile(
    free_disk_gb: float,
    deps_ok: bool,
    logical_cores: int,
    torch_ok: bool = False,
    mpe_ok: bool = False,
) -> str:
    if free_disk_gb < 2 or not deps_ok or not mpe_ok:
        return "PROFILE_SCAFFOLD_ONLY"
    if free_disk_gb < 4:
        return "PROFILE_ULTRA_LEAN"
    if free_disk_gb >= 8 and logical_cores >= 4 and torch_ok:
        return "PROFILE_FULL_SAFE"
    return "PROFILE_LEAN_SAFE"


def budget_mb_for_profile(profile: str) -> int:
    return PROFILE_BUDGET_MB[profile]
