#!/usr/bin/env python
from src.analysis.aggregate import aggregate_results
from src.analysis.tables import make_tables
from pathlib import Path
import csv, json

rows=aggregate_results('results')
make_tables(rows,'latex/tables')
agg=Path('results/aggregate'); agg.mkdir(parents=True,exist_ok=True)
if rows:
    keys=list(rows[0].keys())
    for name in ['training_summary.csv','crossplay_summary.csv','sensitivity_summary.csv']:
        with open(agg/name,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(agg/'resource_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','wall_clock_runtime_sec','artifact_footprint_mb']); w.writeheader();
        for r in rows: w.writerow({k:r[k] for k in ['method','wall_clock_runtime_sec','artifact_footprint_mb']})
    (agg/'final_metrics_for_paper.json').write_text(json.dumps(rows,indent=2))
print('tables done')
