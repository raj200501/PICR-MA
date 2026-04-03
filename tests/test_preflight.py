from src.utils.resource_guard import select_profile

def test_profile_selection_scaffold():
    assert select_profile(1.5, False, 8)=='PROFILE_SCAFFOLD_ONLY'
