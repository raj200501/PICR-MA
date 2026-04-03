#!/usr/bin/env python
from src.analysis.aggregate import aggregate_results
from src.analysis.plots import make_basic_plots
rows=aggregate_results('results')
make_basic_plots(rows,'latex/figures')
print('plots done')
