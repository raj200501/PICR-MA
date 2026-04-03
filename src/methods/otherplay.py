def role_permutation(seed:int, n_agents:int):
    import numpy as np
    return np.random.default_rng(seed).permutation(n_agents)
