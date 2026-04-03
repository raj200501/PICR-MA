import os, platform, shutil, subprocess
from pathlib import Path

def get_system_snapshot():
    total, used, free = shutil.disk_usage('.')
    return {
      'os': platform.platform(),
      'python': platform.python_version(),
      'machine': platform.machine(),
      'apple_silicon': platform.system()=='Darwin' and platform.machine() in {'arm64','aarch64'},
      'logical_cores': os.cpu_count() or 1,
      'free_disk_gb': round(free/1024**3,3),
      'repo_size_mb': round(sum(p.stat().st_size for p in Path('.').rglob('*') if p.is_file())/1024**2,3),
    }
