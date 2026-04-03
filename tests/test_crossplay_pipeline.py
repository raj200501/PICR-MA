from src.eval.crossplay import crossplay_score

def test_crossplay_deterministic():
    assert crossplay_score(1,2)==crossplay_score(1,2)
