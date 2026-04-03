#!/usr/bin/env python
from pathlib import Path
import csv
rows=[]
for p in Path('results').glob('run_*'):
    size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    rows.append((p.name,size/1024**2))
with open('results/artifact_inventory.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['run_id','size_mb']); w.writerows(rows)
print('inventory written')
