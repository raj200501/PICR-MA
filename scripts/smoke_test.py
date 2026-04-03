#!/usr/bin/env python
from scripts.train import run_experiment
run_experiment(env='matrix_coordination',method='ippo',seed=0,steps=50,tag='smoke')
print('smoke ok')
