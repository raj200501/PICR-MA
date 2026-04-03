from src.algos.ppo import reinforce_update

def test_reinforce_update_shape():
    g=reinforce_update([0.1,0.2],1,1.0)
    assert len(g)==2
