from src.utils.resource_guard import select_profile


def test_profile_selection_scaffold():
    assert select_profile(1.5, deps_ok=False, logical_cores=8, mpe_ok=False) == "PROFILE_SCAFFOLD_ONLY"


def test_profile_selection_lean_without_torch():
    assert select_profile(10.0, deps_ok=True, logical_cores=8, torch_ok=False, mpe_ok=True) == "PROFILE_LEAN_SAFE"
