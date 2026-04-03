#!/usr/bin/env python
from pathlib import Path
import importlib, json
from src.utils.system import get_system_snapshot
from src.utils.resource_guard import select_profile

mods = ['torch', 'pettingzoo', 'supersuit', 'gymnasium', 'pygame']
imports = {}
for m in mods:
    try:
        importlib.import_module(m)
        imports[m] = True
    except Exception:
        imports[m] = False

sys = get_system_snapshot()
profile = select_profile(
    sys['free_disk_gb'],
    all(imports.get(k, False) for k in ['torch', 'pettingzoo', 'supersuit']),
    sys['logical_cores'],
)
out = {**sys, 'imports': imports, 'profile': profile, 'battery': 'unknown'}

Path('results').mkdir(exist_ok=True)
Path('results/preflight.json').write_text(json.dumps(out, indent=2))
md = [
    '# PREFLIGHT',
    f"- OS: {out['os']}",
    f"- Python: {out['python']}",
    f"- Apple Silicon: {out['apple_silicon']}",
    f"- Logical cores: {out['logical_cores']}",
    f"- Free disk GB: {out['free_disk_gb']}",
    f"- Repo size MB: {out['repo_size_mb']}",
    f"- Imports: {imports}",
    f"- Selected profile: {profile}",
]
Path('results/PREFLIGHT.md').write_text('\n'.join(md) + '\n')
Path('results/ACTIVE_PROFILE.txt').write_text(profile + '\n')
print(profile)
