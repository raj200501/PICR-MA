from pathlib import Path
import json

def aggregate_results(results_dir='results'):
    rows=[]
    for s in Path(results_dir).glob('run_*/summary.json'):
        rows.append(json.loads(s.read_text()))
    return rows
