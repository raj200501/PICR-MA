import math
from src.envs.observation_slicing import permute_teammates

def _norm(p,eps=1e-8):
    s=sum(max(eps,float(x)) for x in p)
    return [max(eps,float(x))/s for x in p]

def symmetric_kl(p,q,eps=1e-8):
    p=_norm(p,eps); q=_norm(q,eps)
    kl_pq=sum(a*(math.log(a)-math.log(b)) for a,b in zip(p,q))
    kl_qp=sum(b*(math.log(b)-math.log(a)) for a,b in zip(p,q))
    return 0.5*(kl_pq+kl_qp)

def teammate_dropout(blocks,dropout_p,rng):
    out=[]
    for b in blocks:
        out.append([x if rng.random()>dropout_p else 0.0 for x in b])
    return out

def picr_consistency_loss(policy_probs_fn,obs,self_dim,block_dim,n_teammates,rng,dropout_p=0.0):
    perm1=list(range(n_teammates)); perm2=list(range(n_teammates))
    rng.shuffle(perm1); rng.shuffle(perm2)
    o1=permute_teammates(obs,self_dim,block_dim,n_teammates,perm1)
    o2=permute_teammates(obs,self_dim,block_dim,n_teammates,perm2)
    p1=policy_probs_fn([o1])[0]
    p2=policy_probs_fn([o2])[0]
    return float(symmetric_kl(p1,p2))
