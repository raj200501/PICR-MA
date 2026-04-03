from src.envs.observation_slicing import split_self_teammates

def test_split_shapes():
    obs=[0,1,2,3,4,5]
    s,u=split_self_teammates(obs,2,2,2)
    assert len(s)==2 and len(u)==2 and len(u[0])==2
