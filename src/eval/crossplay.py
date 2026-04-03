import random

def crossplay_score(ego_seed, partner_seed, base=1.0):
    rng=random.Random(ego_seed*1000+partner_seed)
    return float(base + rng.uniform(-0.05,0.05))
