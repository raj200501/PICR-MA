from src.models.mlp_policy import TinyPolicy

def test_policy_probs_sum_one():
    pol=TinyPolicy(4,2,0)
    p=pol.probs([[0,0,0,0]])[0]
    assert abs(sum(p)-1)<1e-6
