from src.eval.permutation_sensitivity import sensitivity

def test_sensitivity_nonnegative():
    obs=[1,0,0,1]
    policy=lambda x: [[0.6,0.4] for _ in range(len(x))]
    permute=lambda o,p:o
    assert sensitivity(policy,obs,permute,[None])>=0
