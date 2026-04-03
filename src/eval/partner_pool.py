from pathlib import Path
import json

def list_partners(results_dir='results'):
    partners=[]
    for p in Path(results_dir).glob('run_*'):
        s=p/'summary.json'
        if s.exists():
            partners.append(str(s))
    return partners
