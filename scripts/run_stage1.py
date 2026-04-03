#!/usr/bin/env python
from scripts.train import run_experiment
for env in ['matrix_coordination','matrix_staghunt']:
    for method in ['ippo','picr_ma','deepsets','otherplay']:
        for seed in [0,1]:
            run_experiment(env=env,method=method,seed=seed,steps=200,tag='stage1')
print('stage1 done')
