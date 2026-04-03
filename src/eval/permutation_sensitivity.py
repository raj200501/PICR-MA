from src.utils.metrics_common import js_divergence

def sensitivity(policy_probs_fn,obs,permute_fn,perms):
    p=policy_probs_fn([obs])[0]
    vals=[]
    for perm in perms:
        q=policy_probs_fn([permute_fn(obs,perm)])[0]
        vals.append(js_divergence(p,q))
    return sum(vals)/len(vals)
