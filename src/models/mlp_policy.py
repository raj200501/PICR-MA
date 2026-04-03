import random

class TinyPolicy:
    def __init__(self,obs_dim,act_dim=2,seed=0):
        self.obs_dim=obs_dim; self.act_dim=act_dim
        rng=random.Random(seed)
        self.w=[[rng.uniform(-0.1,0.1) for _ in range(act_dim)] for _ in range(obs_dim)]
    def logits(self,obs):
        out=[]
        for row in obs:
            lg=[0.0 for _ in range(self.act_dim)]
            for i,x in enumerate(row):
                for a in range(self.act_dim):
                    lg[a]+=x*self.w[i][a]
            out.append(lg)
        return out
    def probs(self,obs):
        import math
        all_probs=[]
        for lg in self.logits(obs):
            m=max(lg)
            ex=[math.exp(v-m) for v in lg]
            s=sum(ex)
            all_probs.append([e/s for e in ex])
        return all_probs
