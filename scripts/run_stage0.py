#!/usr/bin/env python
from scripts.train import run_experiment
for m in ['ippo','picr_ma']:
    run_experiment(env='matrix_coordination',method=m,seed=0,steps=80,tag='stage0')
print('stage0 done')
