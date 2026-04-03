def reinforce_update(logits, action, reward, lr=0.05):
    import math
    m=max(logits)
    ex=[math.exp(v-m) for v in logits]
    s=sum(ex)
    probs=[e/s for e in ex]
    grad=[-p for p in probs]
    grad[action]+=1.0
    return [lr*reward*g for g in grad]
