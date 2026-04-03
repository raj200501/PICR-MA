#!/usr/bin/env python
from pathlib import Path
profile=Path('results/ACTIVE_PROFILE.txt').read_text().strip() if Path('results/ACTIVE_PROFILE.txt').exists() else 'UNKNOWN'
print({'profile':profile,'runs':len(list(Path('results').glob('run_*')))})
