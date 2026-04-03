#!/usr/bin/env python
import importlib, json
from pathlib import Path
mods=['pytest','torch','pettingzoo','supersuit','gymnasium']
status={}
for m in mods:
    try:
        importlib.import_module(m); status[m]='ok'
    except Exception as e:
        status[m]=f'missing:{type(e).__name__}'
Path('results').mkdir(exist_ok=True)
Path('results/install_check.json').write_text(json.dumps(status,indent=2))
print(status)
