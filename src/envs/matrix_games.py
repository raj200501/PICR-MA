import random

GAMES={
 'matrix_coordination': [[1.0,0.0],[0.0,1.0]],
 'matrix_staghunt': [[1.0,-0.2],[0.8,0.6]],
}

def softmax(logits):
    import math
    m=max(logits)
    ex=[math.exp(x-m) for x in logits]
    s=sum(ex)
    return [x/s for x in ex]

def sample_action(probs, rng):
    r=rng.random(); c=0.0
    for i,p in enumerate(probs):
        c+=p
        if r<=c:
            return i
    return len(probs)-1

def play_episode(policy_logits, game='matrix_coordination', rng=None):
    rng = random.Random() if rng is None else rng
    probs=softmax(policy_logits)
    a1=sample_action(probs,rng); a2=sample_action(probs,rng)
    r=float(GAMES[game][a1][a2])
    return a1,a2,r
