from .budget import PROFILE_BUDGET_MB

def select_profile(free_disk_gb:float, deps_ok:bool, logical_cores:int)->str:
    if free_disk_gb < 2 or not deps_ok:
        return 'PROFILE_SCAFFOLD_ONLY'
    if free_disk_gb < 4:
        return 'PROFILE_ULTRA_LEAN'
    if free_disk_gb < 8 or logical_cores < 4:
        return 'PROFILE_LEAN_SAFE'
    return 'PROFILE_FULL_SAFE'
