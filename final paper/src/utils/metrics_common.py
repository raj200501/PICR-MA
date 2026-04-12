import math

def _normalize(v, eps=1e-8):
    s=sum(max(eps,float(x)) for x in v)
    return [max(eps,float(x))/s for x in v]

def js_divergence(p,q,eps=1e-8):
    p=_normalize(p,eps); q=_normalize(q,eps)
    m=[0.5*(a+b) for a,b in zip(p,q)]
    def kl(a,b):
        return sum(ai*math.log(ai/bi) for ai,bi in zip(a,b))
    return 0.5*kl(p,m)+0.5*kl(q,m)
